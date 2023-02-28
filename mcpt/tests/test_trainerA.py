from torch import nn
from transformers import AutoModel

from contrastlearning import TrainerA
from contrastlearning import WeightedCosineSimilarityLoss
from .test_base import TestBase


class TestTrainerA(TestBase):
    def __init__(self, name='TestTrainerA'):
        super().__init__(name)

    def test_init(self):
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        N_CLASSES = 14
        head = nn.Sequential(
            nn.Linear(384, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, N_CLASSES),
        )
        trainer = TrainerA(
            model=model,
            head=head,
            device='cpu',
            head_loss=nn.BCEWithLogitsLoss(),
            model_loss=WeightedCosineSimilarityLoss(N_CLASSES),
            model_dataset=None,
            head_dataset=None,
            eval_dataset=None,
            n_classes=N_CLASSES,
            model_loader_type='contrastive',
            unlabeled_dataset=None,
            train_head_batch_size=50,
            head_lr=1e-3,
            model_lr=1e-3,
            head_gamma=.96,
            model_gamma=.96,
            validate_every_n_epochs=2,
            checkpoint_every_n_epochs=5,
        )


def main():
    TestTrainerA().run_testcases()


if __name__ == "__main__":
    main()
