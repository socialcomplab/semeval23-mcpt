import numpy as np
import torch
from torch.utils.data import Sampler
from typing import List, Iterator


class ContrastSampler(Sampler[List[int]]):
    def __init__(self, data_source, batch_size, min_samples_from_class=2, seed=None):
        """Samples batches with at least 'min_samples_from_class' elements from every class.

        :data_source (Dataset): dataset to sample from
        :batch_size (int): batch size
        :min_samples_from_class (int): minimum number of samples to sample per label per batch
        :seed (int): seed for random generator
        """
        self._labels = data_source['labels'].clone().detach().t()
        self._num_classes = self._labels.shape[0]
        if batch_size < self._num_classes * min_samples_from_class:
            raise ValueError("batch size must be larger or equal than num_classes * min_samples_from_class")

        self._hparams = {
            'batch_size': batch_size,
            'min_samples_from_class': min_samples_from_class,
        }
        self._len = len(data_source)
        self._class_idxs = [torch.argwhere(self._labels[i]).flatten() for i in range(self._num_classes)]
        min_num_samples = np.min([len(class_idxs) for class_idxs in self._class_idxs])
        if min_num_samples < min_samples_from_class:
            raise ValueError("min_samples_from_class must be less or equal than the minimum number of samples "
                             f"per label ({min_num_samples})")
        self._is_used = torch.zeros(self._len)
        if not seed:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

    def __iter__(self) -> Iterator[List[int]]:
        while (self._is_used == 0).any():
            idxs = list()
            samples_required = torch.ones(self._num_classes) * self._hparams['min_samples_from_class']
            samples_in_current_batch = torch.zeros(self._len)

            order = torch.randperm(self._num_classes, generator=self._generator)
            for i in order:
                num_to_sample = int(samples_required[i])
                if num_to_sample < 1:
                    continue

                least_used_idxs = self.__get_least_used_idxs_not_in_batch(
                        samples_in_current_batch, self._class_idxs[i], num_to_sample)

                select = torch.randperm(len(least_used_idxs), generator=self._generator)[:num_to_sample]
                selected_idxs = self._class_idxs[i][least_used_idxs[select]]
                self._is_used[selected_idxs] += 1
                samples_in_current_batch[selected_idxs] = 2**20
                idxs.append(selected_idxs)
                samples_required = samples_required - self._labels.t()[selected_idxs].sum(axis=0)

            samples = torch.cat(idxs).tolist()
            num_to_sample = self._hparams['batch_size'] - len(samples)
            least_used_idxs = self.__get_least_used_idxs_not_in_batch(
                    samples_in_current_batch, slice(self._len), num_to_sample)

            select = torch.randperm(len(least_used_idxs), generator=self._generator)[:num_to_sample]
            selected_idxs = least_used_idxs[select]
            self._is_used[selected_idxs] += 1
            samples = samples + selected_idxs.tolist()

            yield samples
        self._is_used = torch.zeros(self._len)

    def __len__(self) -> int:
        return self._len // self._hparams['batch_size']

    def __str__(self) -> str:
        samples_seen_at_least_once = (self._is_used > 0).sum()
        samples_seen_more_than_once = (self._is_used > 1).sum()
        return f'{__class__}\nSeen {samples_seen_at_least_once} samples out of {self._len}\nSeen {samples_seen_more_than_once} samples more than once.'

    def __get_least_used_idxs_not_in_batch(self, samples_in_current_batch, class_idxs, num_to_sample):
        class_uses = self._is_used[class_idxs]
        samples_in_current_batch = samples_in_current_batch[class_idxs]
        class_uses = class_uses + samples_in_current_batch  # do not select any samples already in current batch
        least_uses = torch.min(class_uses)
        least_used_idxs = torch.argwhere(class_uses == least_uses).flatten()
        if len(least_used_idxs) < num_to_sample:
            second_least_used_idxs = torch.argwhere(class_uses == least_uses+1).flatten()
            least_used_idxs = torch.cat([least_used_idxs, second_least_used_idxs])
        return least_used_idxs

    def get_hparams(self):
        return self._hparams
