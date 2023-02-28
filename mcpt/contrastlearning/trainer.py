import datasets
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MultilabelF1Score
from tqdm.auto import trange
from typing import Tuple


class Trainer:
    def __init__(
        self,
        model,
        head,
        device,
        head_loss,
        model_loss,
        head_optimizer,
        model_optimizer,
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
        head_gamma=0.96,
        model_gamma=0.8,
        validate_every_n_epochs=2,
        checkpoint_every_n_epochs=5,
        earliest_checkpoint=50,
    ):
        self.device = device
        self.bert = model
        self.bert.to(device)
        self.head = head
        self.head.to(device)
        self.head_optimizer = head_optimizer
        self.head_scheduler = ExponentialLR(self.head_optimizer, gamma=head_gamma)
        self.head_loss = head_loss
        self.model_optimizer = model_optimizer
        self.model_scheduler = ExponentialLR(self.model_optimizer, gamma=model_gamma)
        self.model_loss = model_loss
        self.model_dataset = model_dataset
        self.head_dataset = head_dataset
        self.eval_dataset = eval_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.hparams = {
            'head_gamma': head_gamma,
            'model_gamma': model_gamma,
            'hbs': train_head_batch_size,
            'mbs': train_model_batch_size,
            'ebs': eval_batch_size,
            'model_loader_type': model_loader_type,
            'num_unsupervised': num_unsupervised,
        }
        self.metrics = {
            "microf1": MultilabelF1Score(num_labels=n_classes, average='micro').to(device),
            "macrof1": MultilabelF1Score(num_labels=n_classes, average='macro').to(device),
        }
        self.log_dict = dict()
        self.validate_every_n_epochs = validate_every_n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.earliest_checkpoint = earliest_checkpoint

    def train_embeddings(self, n_epochs: int) -> None:
        losses = list()
        rng = np.random.default_rng(seed=42)
        for epoch in range(n_epochs):
            self.bert.eval()
            with torch.no_grad():
                if self.unlabeled_dataset:
                    choice = rng.choice(
                            np.arange(stop=len(self.unlabeled_dataset)),
                            size=self.hparams['num_unsupervised'],
                            replace=False)
                    unlabeled_dataset = datasets.Dataset.from_dict(self.unlabeled_dataset[choice]).with_format('torch')
                    tmp1 = self.compute_embeddings(self.model_dataset)
                    tmp2 = self.compute_embeddings(unlabeled_dataset)
                    new_labels = self._label_unlabeled_samples(tmp1, tmp2)
                    train_dataset = datasets.Dataset.from_dict({
                        'input_ids': torch.cat((
                            unlabeled_dataset['input_ids'],
                            self.model_dataset['input_ids'])),
                        'attention_mask': torch.cat((
                            unlabeled_dataset['attention_mask'],
                            self.model_dataset['attention_mask'])),
                        'labels': torch.cat((
                            new_labels,
                            self.model_dataset['labels']))
                    }).with_format('torch')
                else:
                    train_dataset = self.model_dataset
            model_loader = self._get_dataloader(train_dataset, self.hparams['model_loader_type'])
            self.bert.train()
            for batch in model_loader:
                batch = [v.to(self.device) for v in batch.values()]
                loss = self._train_embeddings_step(batch)
                losses.append(loss.detach())
                loss.backward()
                self.model_optimizer.step()
                self.model_optimizer.zero_grad()
            self.log('WCSL', torch.mean(torch.tensor(losses)))
            self.model_scheduler.step()
            self.check_epoch_and_save_checkpoint(epoch, losses, 'finetuning')

    def train_head(self, n_epochs: int) -> None:
        train_embeddings = self.compute_embeddings(self.head_dataset)
        eval_embeddings = None
        if self.validate_every_n_epochs != -1:
            eval_embeddings = self.compute_embeddings(self.eval_dataset)
        train_loader = DataLoader(train_embeddings, self.hparams['hbs'])
        losses = list()
        for epoch in range(n_epochs):
            for batch in train_loader:
                batch = [x.to(self.device) for x in batch]
                loss = self._train_head_step(batch)
                losses.append(loss.detach())
                loss.backward()
                self.head_optimizer.step()
                self.head_optimizer.zero_grad()
            self.log('BCEloss', torch.mean(torch.tensor(losses)))
            self.head_scheduler.step()
            self.check_epoch_and_perform_evaluation(epoch, train_embeddings, eval_embeddings)
        if self.validate_every_n_epochs != -1:
            knearest_logits = self.compute_knearest_prediction(
                train_embeddings.tensors[0],
                eval_embeddings.tensors[0],
                train_embeddings.tensors[1],
            )
            self.log('KNNlogits', knearest_logits)

    def train_hybrid(self, n_epochs: int) -> None:
        losses = list()
        model_loader = self._get_dataloader(self.model_dataset, self.hparams['model_loader_type'])
        for epoch in trange(n_epochs, desc="Epoch"):
            self.bert.train()
            self.head.train()
            for batch in model_loader:
                batch = [v.to(self.device) for v in batch.values()]
                loss = self._train_hybrid_step(batch)
                losses.append(loss.detach())
                loss.backward()
                self.hybrid_optimizer.step()
                self.hybrid_optimizer.zero_grad()
            self.log("Hybridloss", torch.mean(torch.tensor(losses)))
            self.model_scheduler.step()
            if self.validate_every_n_epochs != -1:
                self.bert.eval()
                train_embeddings = self.compute_embeddings(self.head_dataset)
                eval_embeddings = self.compute_embeddings(self.eval_dataset)
                self.check_epoch_and_perform_evaluation(
                    epoch, train_embeddings, eval_embeddings
                )

    def train_joint(self, n_epochs: int) -> None:
        model_loader = self._get_dataloader(self.model_dataset, self.hparams['model_loader_type'])
        for epoch in trange(n_epochs, desc='Epoch'):
            self.bert.train()
            self.head.train()
            torch.set_grad_enabled(True)
            losses = list()
            contrast_losses = list()
            for batch in model_loader:
                batch = [v.to(self.device) for v in batch.values()]
                loss, contrast_loss = self._train_joint_step(batch)
                losses.append(loss.detach())
                contrast_losses.append(contrast_loss.detach())
                loss = contrast_loss + loss
                loss.backward()
                self.model_optimizer.step()
                self.model_optimizer.zero_grad()
                self.head_optimizer.step()
                self.head_optimizer.zero_grad()
            self.model_scheduler.step()
            self.head_scheduler.step()
            self.log('BCEloss', torch.mean(torch.tensor(losses)))
            self.log('WCSL', torch.mean(torch.tensor(contrast_losses)))
            self.bert.eval()
            if self.validate_every_n_epochs != -1 and not epoch % self.validate_every_n_epochs:
                train_embeddings = self.compute_embeddings(self.head_dataset)
                eval_embeddings = self.compute_embeddings(self.eval_dataset)
                knearest_logits = self.compute_knearest_prediction(
                    train_embeddings.tensors[0],
                    eval_embeddings.tensors[0],
                    train_embeddings.tensors[1],
                )
                self.log('KNNlogits', knearest_logits)
                self.check_epoch_and_perform_evaluation(
                    epoch, train_embeddings, eval_embeddings
                )
            self.check_epoch_and_save_checkpoint(epoch, losses, 'joint')

    @torch.no_grad()
    def compute_knearest_prediction(self, train_embeddings: torch.tensor, eval_embeddings: torch.tensor,
                                    train_labels: torch.tensor, temperature=.1):
        dists = torch.cdist(eval_embeddings, train_embeddings)
        weights = torch.exp(-dists / temperature)
        weights = weights / weights.sum(1).unsqueeze(-1).expand(weights.shape)
        nearest_neighbors = torch.argsort(dists)
        nearest_weights = torch.gather(weights, 1, nearest_neighbors)
        nearest_labels = train_labels[nearest_neighbors]
        knearest_logits = (nearest_weights.unsqueeze(-1).expand(nearest_labels.shape) * nearest_labels).sum(1)
        return knearest_logits

    @torch.no_grad()
    def check_epoch_and_perform_evaluation(self, epoch: int, train_embeddings: TensorDataset,
                                           eval_embeddings: TensorDataset) -> None:
        """Performs evaluations on the train and test datasets and logs them to 'log_dict'.

        args:
            epoch: int
            eval_dataset: TensorDataset(embeddings, labels)
            train_dataset: TensorDataset(embeddings, labels)
        """
        if self.validate_every_n_epochs == -1:
            return
        if not epoch % self.validate_every_n_epochs:
            self.bert.eval()
            self.head.eval()
            computed_metrics = self.compute_metrics(train_embeddings, on_eval_dataset=False)
            for metric_name, metric in computed_metrics.items():
                self.log(metric_name, metric)
            computed_metrics = self.compute_metrics(eval_embeddings, on_eval_dataset=True)
            for metric_name, metric in computed_metrics.items():
                self.log(metric_name, metric)

    @torch.no_grad()
    def compute_metrics(self, dataset: TensorDataset, on_eval_dataset: bool = True) -> dict:
        """Returns a 'dict[metric_name] = computed_metric' for every metric in 'self.metrics' evaluated on the eval
        dataset if 'on_eval_dataset=True' otherwise on the train dataset.

        args:
            dataset: TensorDataset(embeddings, labels)
            on_eval_dataset: dataset passed is the eval dataset"""
        prefix = '' if on_eval_dataset else 'train_'
        computed_metrics = dict()
        references = dataset.tensors[1].clone().detach().to(self.device)
        predictions = self.predict(dataset.tensors[0].clone().detach(), self.device)
        computed_metrics[f'{prefix}predictions'] = predictions
        for name, metric in self.metrics.items():
            computed_metrics[f'{prefix}{name}'] = metric(predictions, references).tolist()
        return computed_metrics

    @torch.no_grad()
    def predict(self, embeddings: torch.Tensor, device: str) -> torch.Tensor:
        """Returns predictions and ground truth labels of all samples in the provided dataset and sends them all to
        'device'.

        args:
            embeddings: torch.Tensor
            device"""
        self.head.eval()
        loader = DataLoader(embeddings, batch_size=self.hparams['ebs'])
        predictions = list()
        for batch in loader:
            embeddings = batch.to(self.device)
            prediction_probs = torch.sigmoid(self.head(embeddings))
            tmp_predictions = torch.round(prediction_probs)
            if torch.any(tmp_predictions.sum(1) == 0):
                all_zero_preds = torch.nonzero(tmp_predictions.sum(1) == 0).flatten()
                max_idxs = torch.argmax(prediction_probs[tmp_predictions.sum(1) == 0], dim=1)
                for max_idx, zero_idx in zip(max_idxs, all_zero_preds):
                    tmp_predictions[zero_idx][max_idx] = 1.
            predictions.append(tmp_predictions)
        predictions = torch.cat(predictions).to(device)
        return predictions

    @torch.no_grad()
    def compute_embeddings(self, dataset: datasets.Dataset) -> TensorDataset:
        """Computes sentence embeddings using the current model and returns a 'TensorDataset' containing the embeddings
        and the corresponding labels.

        args:
            dataset: datasets.Dataset(input_ids, attention_mask, labels)"""
        self.bert.eval()
        embeddings = list()
        labels = []
        dataloader = DataLoader(dataset, batch_size=self.hparams['mbs'])
        for batch in dataloader:
            batch = [v.to(self.device) for v in batch.values()]
            input_ids, attention_mask, ls = batch
            tmp_embeddings = self._compute_embeddings_batch(input_ids, attention_mask)
            embeddings.append(tmp_embeddings)
            labels.append(ls)
        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)
        return TensorDataset(embeddings, labels)

    @torch.no_grad()
    def compute_embeddings_unlabeled(self, dataset) -> torch.Tensor:
        self.bert.eval()
        embeddings = list()
        dataloader = DataLoader(dataset, batch_size=self.hparams['mbs'])
        for batch in dataloader:
            batch = [v.to(self.device) for v in batch.values()]
            input_ids, attention_mask = batch
            tmp_embeddings = self._compute_embeddings_batch(input_ids, attention_mask)
            embeddings.append(tmp_embeddings)
        embeddings = torch.cat(embeddings)
        return embeddings

    def _label_unlabeled_samples(self, dataset: TensorDataset, unlabeled_dataset: TensorDataset):
        """Computes the pairwise similarity matrix of all samples in dataset and unlabeled dataset and assigns
        samples in unlabeled_dataset the labels of the sample in dataset with the maximum cosine similarity.

        args:
            dataset (TensorDataset): dataset with tensors 'embeddings' and 'labels'
            unlabeled_dataset (TensorDataset): dataset with first tensor 'embeddings'"""
        logits = self.compute_knearest_prediction(dataset.tensors[0], unlabeled_dataset.tensors[0], dataset.tensors[1])
        new_labels = torch.round(logits)
        #  t1 = F.normalize(dataset.tensors[0], dim=1)
        #  t2 = F.normalize(unlabeled_dataset.tensors[0], dim=1)
        #  similarity = t1.matmul(t2.t())
        #  arg_maxs = torch.max(similarity, dim=0)[1]
        #  new_labels = dataset.tensors[1][arg_maxs]
        return new_labels

    def set_head(self, head, head_optimizer):
        self.head = head
        self.head.to(self.device)
        self.head_optimizer = head_optimizer
        self.head_scheduler = ExponentialLR(self.head_optimizer, gamma=self.hparams['head_gamma'])

    def _train_head_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform one train step on the head and return the loss.

        Override this method.
        """
        raise NotImplementedError

    def _compute_embeddings_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute the sentence embeddings on one batch of tokenized data.

        Override this method.
        """
        raise NotImplementedError

    def _train_embeddings_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute one train step on the body and return the loss.

        Override this method.
        """
        raise NotImplementedError

    def _train_hybrid_step(self, batch) -> torch.Tensor:
        """Perform one train step on the head and body (siamese method) and return the loss.

        Override this method.
        """
        raise NotImplementedError

    def _train_joint_step(self, batch) -> (torch.Tensor, torch.Tensor):
        """Perform one train step on the head and body and return a tuple of loss and contrast loss.

        Override this method.
        """
        raise NotImplementedError

    def _get_dataloader(self, dataset: datasets.Dataset, loader_type: str) -> DataLoader:
        """Override this method."""
        raise NotImplementedError

    @staticmethod
    def plot_metrics(log_dict: dict, validate_every: int) -> None:
        """args:
            log_dict: dict of metrics containing key 'BCEloss', 'microf1', 'macrof1', 'train_microf1', 'train_macrof1'
            validate_every: interval of evaluations
            """
        n_epochs = len(log_dict['BCEloss'])
        fig, ax = plt.subplots()
        # loss1 = ax.plot(np.arange(n_epochs), np.array(log_dict['WCSloss']), label='WCS loss')
        loss = ax.plot(np.arange(n_epochs), log_dict['BCEloss'], label='BCE loss', c='green')
        ax2 = ax.twinx()
        x_axis = np.arange(start=validate_every, stop=n_epochs+validate_every, step=validate_every)
        lns1 = ax2.plot(x_axis, log_dict['microf1'], label="Micro F1", c='red')
        lns2 = ax2.plot(x_axis, log_dict['macrof1'], label="Macro F1", c='orange')
        lns3 = ax2.plot(x_axis, log_dict['train_microf1'], label="Train Micro F1", c='red', linestyle='dotted')
        lns4 = ax2.plot(x_axis, log_dict['train_macrof1'], label="Train Macro F1", c='orange', linestyle='dotted')
        ax2.axhline(np.max(log_dict['microf1']), c='red', linestyle='dashed')
        ax2.axhline(np.max(log_dict['macrof1']), c='orange', linestyle='dashed')
        lns = loss+lns1+lns2+lns3+lns4
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs)
        plt.show()
        argmax_microf1 = np.argmax(log_dict['microf1'])
        print("MicroF1: ", np.max(log_dict['microf1']), " @ ", argmax_microf1 * validate_every)
        print("MacroF1: ", np.max(log_dict['macrof1']), " @ ", np.argmax(log_dict['macrof1']) * validate_every)
        print("MacroF1: ", log_dict['macrof1'][argmax_microf1], " @ ", argmax_microf1 * validate_every)

    def log(self, metric_name: str, value) -> None:
        if metric_name not in self.log_dict:
            self.log_dict[metric_name] = [value]
        else:
            self.log_dict[metric_name].append(value)

    def check_epoch_and_save_checkpoint(self, epoch: int, losses: list, ckpt_name: str):
        if epoch > self.earliest_checkpoint and not (epoch+1) % self.checkpoint_every_n_epochs:
            loss = 0 if not len(losses) else losses[-1]
            self.save_checkpoint(f'{ckpt_name}_{epoch+1}', epoch, loss)

    def save_checkpoint(self, ckpt_name: str, epoch: int = 0, loss: float = .0):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.bert.state_dict(),
            'head_state_dict': self.head.state_dict(),
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'model_scheduler_state_dict': self.model_scheduler.state_dict(),
            'head_optimizer_state_dict': self.head_optimizer.state_dict(),
            'head_scheduler_state_dict': self.head_scheduler.state_dict(),
            'loss': loss,
        }, f'{ckpt_name}.ckpt')

    def load_from_checkpoint(self, ckpt_name: str) -> Tuple[int, float]:
        checkpoint = torch.load(f'{ckpt_name}.ckpt')
        self.bert.load_state_dict(checkpoint['model_state_dict'])
        self.head.load_state_dict(checkpoint['head_state_dict'])
        self.model_optimizer.load_state_dict(checkpoint['model_optimizer_state_dict'])
        self.model_scheduler.load_state_dict(checkpoint['model_scheduler_state_dict'])
        self.head_optimizer.load_state_dict(checkpoint['head_optimizer_state_dict'])
        self.head_scheduler.load_state_dict(checkpoint['head_scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

    def save_hparams(self, filename: str = 'hparams'):
        with open(f'{filename}.pickle', 'wb') as f:
            pickle.dump(self.hparams, f)

    def load_hparams(self, filename):
        with open(f'{filename}.pickle', 'rb') as f:
            self.hparams = pickle.load(f)
        for g in self.model_optimizer.param_groups:
            g['lr'] = self.hparams['model_lr']
        for g in self.head_optimizer.param_groups:
            g['lr'] = self.hparams['head_lr']
        self.model_scheduler.gamma = self.hparams['model_gamma']
        self.head_scheduler.gamma = self.hparams['head_gamma']

    def save_log_dict(self, filename='logdict'):
        with open(f'{filename}.pickle', 'wb') as f:
            pickle.dump(self.log_dict, f)
