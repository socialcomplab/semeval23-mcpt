import numpy as np
import torch
from datasets import Dataset

from contrastlearning import ContrastSampler
from .test_base import TestBase


class TestSampler(TestBase):
    def __init__(self, name='TestSampler'):
        super().__init__(name)

    def test_sampler_exceptions(self):
        labels = torch.tensor([
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
        ])
        dataset = Dataset.from_dict({"labels": labels}).with_format("torch")
        try:
            sampler = ContrastSampler(dataset, batch_size=1, min_samples_from_class=2)
            assert 'Expected ValueError exception' and False
        except ValueError as e:
            assert str(e).startswith('batch size must be larger or equal than')
        try:
            sampler = ContrastSampler(dataset, batch_size=6, min_samples_from_class=2)
            assert 'Expected ValueError exception' and False
        except ValueError as e:
            assert str(e).startswith('min_samples_from_class must be less or equal than the minimum number of samples')

    def test_sampler_values(self):
        labels = torch.randint(low=0, high=2, size=(270, 14))
        random_tensor = torch.zeros(size=(270, 1))
        dataset = Dataset.from_dict({"labels": labels, "random": random_tensor}).with_format("torch")
        sampler = ContrastSampler(dataset, batch_size=51)
        assert sampler._num_classes == 14
        assert len(sampler) == 5
        assert sampler._len == 270

    def test_sampler(self):
        labels = torch.tensor([
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0],
        ])
        dataset = Dataset.from_dict({"labels": labels}).with_format("torch")
        sampler = ContrastSampler(dataset, batch_size=6, min_samples_from_class=2, seed=42)
        batches = list()
        for batch in sampler:
            batches.append(batch)
        batches = np.array(batches)
        _, counts = np.unique(batches, return_counts=True)
        assert counts[0] == 6
        assert counts[1] == 6
        assert len(counts) == len(labels)


def main():
    TestSampler().run_testcases()


if __name__ == "__main__":
    main()
