import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import RandomSampler

from .trainer import Trainer
from .sampler import ContrastSampler


class TrainerA(Trainer):
    def __init__(
        self,
        model,
        head,
        device,
        head_loss,
        model_loss,
        model_dataset,
        head_dataset,
        eval_dataset,
        n_classes,
        model_loader_type,
        unlabeled_dataset=None,
        num_unsupervised=500,
        train_head_batch_size=100,
        train_model_batch_size=50,
        eval_batch_size=200,
        head_lr=1e-3,
        head_gamma=0.96,
        model_lr=1e-5,
        model_gamma=0.8,
        beta=0.01,
        min_samples_from_class=2,
        validate_every_n_epochs=2,
        checkpoint_every_n_epochs=5,
        earliest_checkpoint=50,
    ):
        head_optimizer = AdamW(head.parameters(), lr=head_lr)
        model_optimizer = AdamW(model.parameters(), lr=model_lr)
        super().__init__(model, head, device, head_loss, model_loss, head_optimizer, model_optimizer, model_dataset,
                         head_dataset, eval_dataset, n_classes, model_loader_type, unlabeled_dataset, num_unsupervised,
                         train_head_batch_size, train_model_batch_size, eval_batch_size, head_gamma, model_gamma,
                         validate_every_n_epochs, checkpoint_every_n_epochs, earliest_checkpoint)
        self.hparams['head_lr'] = head_lr
        self.hparams['model_lr'] = model_lr
        self.hparams['beta'] = beta
        self.hparams['min_samples_from_class'] = min_samples_from_class

    def _train_head_step(self, batch):
        sentence_embeddings, labels = batch
        prediction_probs = self.head(sentence_embeddings)
        return self.head_loss(prediction_probs, labels)

    def _train_embeddings_step(self, batch):
        input_ids, attention_mask, labels = batch
        embeddings = self.bert(input_ids, attention_mask)
        sentence_embeddings = self._mean_pooling(embeddings, attention_mask)
        #  sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return self.model_loss(sentence_embeddings, labels)

    def _train_joint_step(self, batch):
        input_ids, attention_mask, labels = batch
        embeddings = self.bert(input_ids, attention_mask)
        sentence_embeddings = self._mean_pooling(embeddings, attention_mask)
        contrast_loss = self.model_loss(sentence_embeddings, labels)
        prediction_probs = self.head(sentence_embeddings)
        loss = self.head_loss(prediction_probs, labels)
        return loss, contrast_loss * self.hparams['beta']

    def _compute_embeddings_batch(self, input_ids, attention_mask):
        embeddings = self.bert(input_ids, attention_mask=attention_mask)
        sentence_embeddings = self._mean_pooling(embeddings, attention_mask)
        #  sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def _get_dataloader(self, dataset, loader_type):
        if loader_type == 'contrast':
            sampler = ContrastSampler(
                dataset,
                self.hparams['mbs'],
                min_samples_from_class=self.hparams['min_samples_from_class'],
                seed=42
            )
            return DataLoader(dataset, batch_sampler=sampler)
        elif loader_type == 'random':
            g_cpu = torch.Generator()
            g_cpu.manual_seed(42)
            sampler = RandomSampler(dataset, generator=g_cpu)
            return DataLoader(dataset, sampler=sampler, batch_size=self.hparams['mbs'])
        raise NotImplementedError

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sentence_embeddings = sum_embeddings / sum_mask

        #  sentence_embeddings_expanded = sentence_embeddings.unsqueeze(1).expand(token_embeddings.size())
        #  diffs = (token_embeddings - sentence_embeddings_expanded) * input_mask_expanded
        #  diffs2 = diffs * diffs
        #  stds = torch.sqrt(torch.sum(diffs2, 1) / (sum_mask - 1)) # Bessel correction
        #  stds[stds == .0] = 1e-9
        #  return torch.clip(sentence_embeddings / stds, -2, 2)
        return sentence_embeddings
