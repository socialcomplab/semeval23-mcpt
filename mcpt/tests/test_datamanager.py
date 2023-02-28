from math import ceil
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from contrastlearning import DataManager, INT2LABEL
from .test_base import TestBase

EN_TRAIN_DATA_LENGTH = 433
EN_DEV_DATA_LENGTH = 83
GE_TRAIN_DATA_LENGTH = 132
GE_DEV_DATA_LENGTH = 45
FR_TRAIN_DATA_LENGTH = 158
IT_TRAIN_DATA_LENGTH = 227


class TestDataManager(TestBase):
    def __init__(self, name="TestDataManager"):
        super().__init__(name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.dm = DataManager(self.tokenizer)

    def test_num_classes(self):
        assert self.dm.num_classes == 14

    def test_contrast_dataset(self):
        contrastive_dataset = self.dm.get_contrastive_dataset()
        required_features = ["input_ids", "attention_mask", "labels"]

        assert len(contrastive_dataset.column_names) == len(
            required_features
        ), "Too many features are present in the dataset"
        assert all(
            feature in contrastive_dataset.column_names for feature in required_features
        ), "Not all features are present in the dataset"
        assert contrastive_dataset.shape[0] == int(
            EN_TRAIN_DATA_LENGTH * 0.8
        ), "The number of rows in the dataset is not correct"
        assert (
            contrastive_dataset["input_ids"].shape[1]
            == contrastive_dataset["attention_mask"].shape[1]
        )
        assert (
            contrastive_dataset["input_ids"].shape[1] == self.tokenizer.model_max_length
        )
        assert contrastive_dataset["labels"].shape[1] == self.dm.num_classes

    def test_head_train(self):
        train_head = self.dm.get_head_train_dataset()
        required_features = ["input_ids", "attention_mask", "labels"]

        assert all(
            feature in train_head.column_names for feature in required_features
        ), "Not all features are present in the dataset"
        assert train_head.shape[0] == int(
            EN_TRAIN_DATA_LENGTH * 0.8
        ), "The number of rows in the dataset is not correct"
        assert train_head["input_ids"].shape[1] == train_head["attention_mask"].shape[1]
        assert train_head["input_ids"].shape[1] == self.tokenizer.model_max_length
        assert train_head["labels"].shape[1] == self.dm.num_classes

    def test_head_eval(self):
        eval_head = self.dm.get_head_eval_dataset()
        required_features = ["input_ids", "attention_mask", "labels"]

        assert all(
            feature in eval_head.column_names for feature in required_features
        ), "Not all features are present in the dataset"
        assert eval_head.shape[0] == ceil(
            EN_TRAIN_DATA_LENGTH * 0.2
        ), "The number of rows in the dataset is not correct"
        assert eval_head["input_ids"].shape[1] == eval_head["attention_mask"].shape[1]
        assert eval_head["input_ids"].shape[1] == self.tokenizer.model_max_length
        assert eval_head["labels"].shape[1] == self.dm.num_classes

    def test_dataset_contrastive_dataloader(self):
        contrastive_dataset = self.dm.get_contrastive_dataset()
        dl = DataLoader(contrastive_dataset, batch_size=3)
        batch = next(iter(dl))
        assert isinstance(batch, dict), "Wrong type for the batch"
        assert len(batch) == 3, "Wrong number of features in the batch"
        assert all(
            feature in batch.keys()
            for feature in ["input_ids", "attention_mask", "labels"]
        )
        assert all(isinstance(batch, torch.Tensor) for batch in batch.values())

    def test_drop_classes(self):
        dm = DataManager(self.tokenizer, drop_classes=INT2LABEL[4:7])
        contrastive_dataset = dm.get_contrastive_dataset()
        head_train_dataset = dm.get_head_train_dataset()
        head_eval_dataset = dm.get_head_eval_dataset()

        print(contrastive_dataset["labels"].shape)
        assert contrastive_dataset["labels"].shape == torch.Size(
            [
                int(EN_TRAIN_DATA_LENGTH * 0.8),
                len(INT2LABEL),
            ]
        ), "The labels are not correct for the contrastive dataset"
        assert (
            torch.count_nonzero(contrastive_dataset["labels"][:, 4:7]) == 0
        ), "The classes were not dropped correctly for the contrastive dataset"
        assert head_train_dataset["labels"].shape == torch.Size(
            [
                int(EN_TRAIN_DATA_LENGTH * 0.8),
                len(INT2LABEL),
            ]
        ), "The labels are not correct for the head train dataset"
        assert (
            torch.count_nonzero(head_train_dataset["labels"][:, 4:7]) == 0
        ), "The classes were not dropped correctly for the head train dataset"
        assert head_eval_dataset["labels"].shape == torch.Size(
            [
                ceil(EN_TRAIN_DATA_LENGTH * 0.2),
                len(INT2LABEL),
            ]
        ), "The labels are not correct for the head eval dataset"
        assert (
            torch.count_nonzero(head_eval_dataset["labels"][:, 4:7]) > 0
        ), "The classes were not dropped correctly for the head eval dataset"

    def test_datamanager_cv(self):
        dm = DataManager(
            self.tokenizer,
            languages_for_contrastive=["en", "ge"],
            languages_for_head_train=["fr"],
            languages_for_head_eval=["ge", "it"],
        )
        allowed_deviation = 2
        for (
            contrastive_dataset,
            head_train_dataset,
            head_eval_dataset,
        ) in dm.cross_validation_iter():
            # english is only used for training, german is also used for evaluation
            expected_contrastive_length = EN_TRAIN_DATA_LENGTH + int(
                GE_TRAIN_DATA_LENGTH * 0.8
            )
            # french is only used for training
            expected_head_train_length = FR_TRAIN_DATA_LENGTH
            # german is also in training data, italian only evaluation
            expected_head_eval_length = (
                ceil(GE_TRAIN_DATA_LENGTH * 0.2) + IT_TRAIN_DATA_LENGTH
            )

            assert (
                -allowed_deviation
                <= contrastive_dataset.shape[0] - expected_contrastive_length
                <= +allowed_deviation
            ), f"The number of rows in the dataset is not correct: expected {expected_contrastive_length}, got {contrastive_dataset.shape[0]}"
            assert (
                -allowed_deviation
                <= head_train_dataset.shape[0] - expected_head_train_length
                <= +allowed_deviation
            ), f"The number of rows in the dataset is not correct: expected {expected_head_train_length}, got {head_train_dataset.shape[0]}"
            assert (
                -allowed_deviation
                <= head_eval_dataset.shape[0] - expected_head_eval_length
                <= +allowed_deviation
            ), f"The number of rows in the dataset is not correct: expected {expected_head_eval_length}, got {head_eval_dataset.shape[0]}"

    def test_datamanager_cv_wrong_usage(self):
        dm = DataManager(
            self.tokenizer,
            languages_for_head_eval=["ge"],
        )
        cv_iter = dm.cross_validation_iter()
        contrastive, _, head_eval = next(cv_iter)
        assert (
            contrastive.shape[0] == EN_TRAIN_DATA_LENGTH
        ), f"The number of rows in the dataset is not correct: expected {EN_TRAIN_DATA_LENGTH}, got {contrastive.shape[0]}"
        assert (
            head_eval.shape[0] == GE_TRAIN_DATA_LENGTH
        ), f"The number of rows in the dataset is not correct: expected {GE_TRAIN_DATA_LENGTH}, got {head_eval.shape[0]}"
        try:
            next(cv_iter)
            assert False, "Should have raised an exception"
        except StopIteration:
            pass

    def test_datamanager_dev(self):
        dm = DataManager(
            self.tokenizer,
            use_dev=True,
        )
        contrastive_dataset = dm.get_contrastive_dataset()
        assert (
            contrastive_dataset.shape[0] == EN_TRAIN_DATA_LENGTH
        ), f"The number of rows in the dataset is not correct: expected {EN_TRAIN_DATA_LENGTH}, got {contrastive_dataset.shape[0]}"
        head_eval = dm.get_head_eval_dataset()
        assert (
            head_eval.shape[0] == EN_DEV_DATA_LENGTH
        ), f"The number of rows in the dataset is not correct: expected {EN_DEV_DATA_LENGTH}, got {head_eval.shape[0]}"

        dm = DataManager(self.tokenizer, use_dev=True, languages_for_head_eval=["ge"])
        contrastive_dataset = dm.get_contrastive_dataset()
        assert (
            contrastive_dataset.shape[0] == EN_TRAIN_DATA_LENGTH + EN_DEV_DATA_LENGTH
        ), f"The number of rows in the dataset is not correct: expected {EN_TRAIN_DATA_LENGTH + EN_DEV_DATA_LENGTH}, got {contrastive_dataset.shape[0]}"
        head_eval = dm.get_head_eval_dataset()
        assert (
            head_eval.shape[0] == GE_TRAIN_DATA_LENGTH + GE_DEV_DATA_LENGTH
        ), f"The number of rows in the dataset is not correct: expected {GE_TRAIN_DATA_LENGTH + GE_DEV_DATA_LENGTH}, got {head_eval.shape[0]}"

    def test_dataset_contrastive_pairs(self):
        dm = DataManager(self.tokenizer, create_contrastive_pairs=True)
        contrastive_dataset = dm.get_contrastive_dataset()

        required_features = ["texts", "labels", "category"]
        assert all(
            column in contrastive_dataset.column_names for column in required_features
        ), "Wrong features in the contrastive dataset"

        dl = DataLoader(
            contrastive_dataset,
            batch_size=5,
            shuffle=True,
            collate_fn=dm.contrastive_pairs_collate_fn,
        )
        batch = next(iter(dl))
        (
            input_ids_lhs,
            attention_mask_lhs,
            labels_lhs,
            input_ids_rhs,
            attention_mask_rhs,
            labels_rhs,
            category,
        ) = batch.values()
        assert (
            input_ids_lhs.shape
            == attention_mask_lhs.shape
            == input_ids_rhs.shape
            == attention_mask_rhs.shape
        ), "Wrong shape of sentence features"
        assert isinstance(labels_lhs, torch.Tensor), "Wrong type for similarity"
        assert isinstance(labels_rhs, torch.Tensor), "Wrong type for similarity"
        assert isinstance(category, torch.Tensor), "Wrong type for category"

        assert (
            labels_lhs.shape == labels_rhs.shape
        ), "Labels shapes don't match"
        assert (
            input_ids_lhs.shape[0] == labels_lhs.shape[0] == category.shape[0]
        ), "Rows in the batch don't match"


def main():
    TestDataManager().run_testcases()


if __name__ == "__main__":
    main()
