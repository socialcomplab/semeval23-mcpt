import torch

from contrastlearning import WeightedCosineSimilarityLoss
from contrastlearning import ContrastLoss2
from .test_base import TestBase


class TestLoss(TestBase):
    def __init__(self, name='TestLoss'):
        super().__init__(name)

    def test_loss_sanity_checks(self):
        n_features = 100
        feature_size = 200
        features = torch.rand((n_features, feature_size))
        losses = list()
        for n_classes in range(20):
            labels = torch.randint(low=0, high=2, size=(n_features, n_classes))
            loss = WeightedCosineSimilarityLoss(n_classes)
            losses.append(loss(features, labels))
        assert losses[0] == .0

    def test_hamming_weight(self):
        loss = WeightedCosineSimilarityLoss(3)

        labels = torch.ones((3, 3))
        result = loss.hamming_distance_by_matrix(labels)
        expected_result = torch.zeros((3, 3))
        assert torch.all(result == expected_result)

        labels = torch.zeros((3, 3))
        result = loss.hamming_distance_by_matrix(labels)
        expected_result = torch.zeros((3, 3))
        assert torch.all(result == expected_result)

        labels = torch.tensor([
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ])
        result = loss.hamming_distance_by_matrix(labels)
        expected_result = torch.tensor([
            [0, 1, 1],
            [1, 0, 2],
            [1, 2, 0],
        ])
        assert torch.all(result == expected_result)


class TestContrastLoss2(TestBase):
    def __init__(self, name='TestContrastLoss2'):
        super().__init__(name)
        self.loss = ContrastLoss2(temp=10)

    def test_loss(self):
        labels = torch.tensor([
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ])
        features = torch.tensor([
            [1, 2, 3],
            [5, 2, 4],
            [3, 4, 2]
        ], dtype=torch.float32)
        loss = self.loss(features, labels)
        assert loss >= .0


def main():
    TestLoss().run_testcases()


if __name__ == "__main__":
    main()
