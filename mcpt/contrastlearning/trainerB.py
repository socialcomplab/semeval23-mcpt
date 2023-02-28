import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch import optim
from tqdm.auto import trange, tqdm

from .trainer import Trainer
from .custom_loss import MultiCategoricalSiameseLoss
from .sampler import ContrastSampler


class TrainerB(Trainer):
    def __init__(
        self,
        transformer,
        model_dataset,
        head_dataset,
        eval_dataset,
        model_batch_size,
        num_classes,
        device,
        model_loader_type="pair",
        normalize_embeddings=False,
        head_lr=1e-3,
        body_lr=1e-5,
        hybrid_lr=1e-5,
        hybrid_contrast_factor=1.0,
        hybrid_head_factor=1.0,
        head_gamma=0.9,
        body_gamma=0.9,
        validate_every_n_epochs=2,
        checkpoint_every_n_epochs=5,
        simple_head=False,
        pairs_collate_fn=None,
        eval_batch_size=100,
    ):
        head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(transformer.get_sentence_embedding_dimension(), num_classes),
            nn.Dropout(),
        )
        body_loss = MultiCategoricalSiameseLoss(
            transformer.get_sentence_embedding_dimension(),
            num_classes,
            simple=simple_head,
        ).to(device)
        head_loss = nn.BCEWithLogitsLoss()
        head_optimizer = optim.AdamW(head.parameters(), lr=head_lr)
        body_optimizer = optim.AdamW(
            list(transformer.parameters()) + list(body_loss.parameters()), lr=body_lr
        )
        self.hybrid_optimizer = optim.AdamW(
            list(transformer.parameters())
            + list(body_loss.parameters())
            + list(head.parameters()),
            lr=hybrid_lr,
        )
        super().__init__(
            model=transformer,
            head=head,
            device=device,
            model_loss=body_loss,
            head_loss=head_loss,
            model_dataset=model_dataset,
            head_dataset=head_dataset,
            eval_dataset=eval_dataset,
            model_loader_type=model_loader_type,
            n_classes=num_classes,
            head_optimizer=head_optimizer,
            model_optimizer=body_optimizer,
            head_gamma=head_gamma,
            model_gamma=body_gamma,
            validate_every_n_epochs=validate_every_n_epochs,
            checkpoint_every_n_epochs=checkpoint_every_n_epochs,
            train_model_batch_size=model_batch_size,
            eval_batch_size=eval_batch_size,
        )

        self.normalize_embeddings = normalize_embeddings
        self.pairs_collate_fn = pairs_collate_fn
        self.hybrid_contrast_factor = hybrid_contrast_factor
        self.hybrid_head_factor = hybrid_head_factor

    def _train_head_step(self, batch):
        sentence_embedding, labels = batch
        prediction = self.head(sentence_embedding)
        return self.head_loss(prediction, labels)

    def _train_hybrid_step(self, batch):
        if len(batch) == 7:
            (
                input_ids_lhs,
                attention_mask_lhs,
                labels_lhs,
                input_ids_rhs,
                attention_mask_rhs,
                labels_rhs,
                categories,
            ) = batch
            sentence_embeddings = [
                self._compute_embeddings_batch(input_ids_lhs, attention_mask_lhs),
                self._compute_embeddings_batch(input_ids_rhs, attention_mask_rhs),
            ]

            similarities = (
                torch.logical_and(
                    labels_lhs.gather(1, categories.unsqueeze(1)),
                    labels_rhs.gather(1, categories.unsqueeze(1)),
                )
                .squeeze(1)
                .float()
            )
            loss = self.model_loss(
                sentence_embeddings[0], sentence_embeddings[1], categories, similarities
            ) * self.hybrid_contrast_factor
            loss += self._train_head_step(
                (
                    sentence_embeddings[0],
                    labels_lhs,
                )
            ) * self.hybrid_head_factor
            loss += self._train_head_step(
                (
                    sentence_embeddings[1],
                    labels_rhs,
                )
            ) * self.hybrid_head_factor
            return loss
        else:
            input_ids, attention_mask, labels = batch
            sentence_embeddings = self._compute_embeddings_batch(
                input_ids, attention_mask
            )
            prediction = self.head(sentence_embeddings)

            loss = self.model_loss.forward_batch(sentence_embeddings, labels)
            # weight based on how often sample was used for embedding training
            head_loss_weight = torch.zeros(labels.shape[0]).to(self.device)
            for batch in self.model_loss.compute_train_pairs(labels):
                first_idx, second_idx = batch["indices"]
                head_loss_weight[first_idx] += torch.tensor(1.0).to(self.device)
                head_loss_weight[second_idx] += torch.tensor(1.0).to(self.device)

            for i, w in enumerate(head_loss_weight):
                loss += self.head_loss(prediction[i], labels[i]) * w
            return loss

    def _get_dataloader(self, dataset, loader_type):
        if loader_type == 'contrast':
            sampler = ContrastSampler(dataset, self.hparams['mbs'], seed=42)
            return DataLoader(dataset, batch_sampler=sampler)
        elif loader_type == "pairs":
            return DataLoader(
                dataset,
                batch_size=self.hparams["mbs"],
                shuffle=True,
                collate_fn=self.pairs_collate_fn,
            )
        elif loader_type == "random":
            sampler = RandomSampler(dataset)
            return DataLoader(dataset, batch_size=self.hparams["mbs"], sampler=sampler)
        raise NotImplementedError

    def _train_embeddings_step(self, batch):
        if len(batch) == 7:
            # pair method
            (
                input_ids_lhs,
                attention_mask_lhs,
                labels_lhs,
                input_ids_rhs,
                attention_mask_rhs,
                labels_rhs,
                categories,
            ) = batch

            sentence_embeddings = [
                self._compute_embeddings_batch(input_ids_lhs, attention_mask_lhs),
                self._compute_embeddings_batch(input_ids_rhs, attention_mask_rhs),
            ]

            similarities = (
                torch.logical_and(
                    labels_lhs.gather(1, categories.unsqueeze(1)),
                    labels_rhs.gather(1, categories.unsqueeze(1)),
                )
                .squeeze(1)
                .float()
            )
            return self.model_loss(
                sentence_embeddings[0], sentence_embeddings[1], categories, similarities
            )
        else:  # batch method
            input_ids, attention_mask, labels = batch
            sentence_embeddings = self._compute_embeddings_batch(
                input_ids, attention_mask
            )
            return self.model_loss.forward_batch(sentence_embeddings, labels)

    def _compute_embeddings_batch(self, input_ids, attention_mask):
        sentence_embeddings = self.bert(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )["sentence_embedding"]
        if self.normalize_embeddings:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
