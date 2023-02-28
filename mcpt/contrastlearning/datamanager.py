from dataclasses import dataclass, field
import glob
from logging import warning
from typing import Iterator, List, Optional, Tuple
import os
import numpy as np

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset
import torch
import datasets
from datasets import load_dataset, concatenate_datasets
from tokenizers import Tokenizer

from .trainer import Trainer

datasets.utils.disable_progress_bar()
datasets.utils.logging.set_verbosity_error()


INT2LABEL = sorted(
    [
        "Legality_Constitutionality_and_jurisprudence",
        "Quality_of_life",
        "Cultural_identity",
        "Fairness_and_equality",
        "Health_and_safety",
        "Policy_prescription_and_evaluation",
        "Political",
        "Capacity_and_resources",
        "Economic",
        "Public_opinion",
        "Morality",
        "Crime_and_punishment",
        "External_regulation_and_reputation",
        "Security_and_defense",
    ]
)
LABEL2INT = {label: i for i, label in enumerate(INT2LABEL)}


@dataclass
class DataManager:
    """Data Manager class

    This class is used to load training data and test data, and create corresponding dataloaders.

    Args:
        tokenizer: Tokenizer object
        data_dir: Path to the data directory with datasets in directory structure
            `{data_dir}/{language}/{'train'/'dev'}-articles-subtask-2/articleXYZ.txt`
            and `{data_dir}/{language}/{'train'/'dev'}-labels-subtask-2.txt`
        use_dev: Whether to use the dev set. Will use it as eval set for languages that are used for evaluation,
            otherwise it will be added to the training set.
    """

    tokenizer: Tokenizer
    data_dir: str = "./data/"

    use_dev: bool = False
    drop_classes: List[str] = field(default_factory=lambda: [])

    languages_for_contrastive: List[str] = field(default_factory=lambda: ["en"])
    languages_for_head_train: List[str] = field(default_factory=lambda: ["en"])
    languages_for_head_eval: List[str] = field(default_factory=lambda: ["en"])

    create_contrastive_pairs: bool = False
    contrastive_weighting_base: float = 1  # 1 means no weighting
    num_contrastive_samples: int = 1

    _train_only_dataset: Optional[Dataset] = None
    _eval_only_dataset: Optional[Dataset] = None
    _train_and_eval_dataset: Optional[Dataset] = None
    _split_datasets: Optional[Tuple[Dataset, Dataset]] = None

    def get_contrastive_dataset(self) -> Dataset:
        contrastive_raw = self._load_languages_datasets(self.languages_for_contrastive)
        return self._preprocess_contrastive_dataset(contrastive_raw)

    def get_head_train_dataset(self) -> Dataset:
        dataset = self._load_languages_datasets(self.languages_for_head_train)
        return self._preprocess_head_dataset(dataset)

    def get_head_eval_dataset(self) -> Dataset:
        dataset = self._load_languages_datasets(
            self.languages_for_head_eval, train=False
        )
        if len(dataset):
            return self._preprocess_head_dataset(dataset, eval=True)
        return []

    def cross_validation_iter(self) -> Iterator[Tuple[Dataset, Dataset, Dataset]]:
        self._load_all_datasets()
        if self._train_and_eval_dataset is None:
            warning(
                "No overlap in eval and train languages, cross-validation will not be performed, returning the default dataset -> using all training languages data for training and all eval data for evaluation."
            )
            yield (
                self.get_contrastive_dataset(),
                self.get_head_train_dataset(),
                self.get_head_eval_dataset(),
            )
            return

        cv = StratifiedKFold()
        for train_indices, eval_indices in cv.split(
            self._train_and_eval_dataset,
            self._create_stratify_column(self._train_and_eval_dataset["language"]),
        ):
            train_dataset = self._train_and_eval_dataset.select(train_indices)
            eval_dataset = self._train_and_eval_dataset.select(eval_indices)

            if self._train_only_dataset is not None:
                train_dataset = concatenate_datasets(
                    [train_dataset, self._train_only_dataset]
                )
            if self._eval_only_dataset is not None:
                eval_dataset = concatenate_datasets(
                    [eval_dataset, self._eval_only_dataset]
                )

            _contrastive_dataset = self._preprocess_contrastive_dataset(
                self._filter_languages(train_dataset, self.languages_for_contrastive)
            )
            _head_train_dataset = self._preprocess_head_dataset(
                self._filter_languages(train_dataset, self.languages_for_head_train)
            )
            _head_eval_dataset = self._preprocess_head_dataset(
                self._filter_languages(eval_dataset, self.languages_for_head_eval),
                eval=True,
            )
            yield _contrastive_dataset, _head_train_dataset, _head_eval_dataset

    def predict_and_write(self, trainer: Trainer, articles_dir: str, output_file: str):
        filenames = glob.glob(f"{articles_dir}/*.txt")
        ids = map(lambda name: name.split('article')[-1][:-len('.txt')], filenames)
        dataset_to_score = _get_features_from_files(filenames)
        dataset_to_score = self.preprocess_predict_dataset(dataset_to_score)
        embeddings = trainer.compute_embeddings_unlabeled(dataset_to_score)
        predictions = trainer.predict(embeddings, trainer.device)
        labels = map(_vec_to_label_names, predictions)
        output_frame = pd.DataFrame({'ids': ids, 'labels': labels})
        output_frame.to_csv(output_file, sep='\t', header=False, index=False)
        return predictions

    def _preprocess_contrastive_dataset(self, contrastive_dataset):
        contrastive_dataset = contrastive_dataset.map(
            self._maybe_drop_classes, batched=True
        )

        if self.create_contrastive_pairs:
            return self._create_contrastive_pairs_dataset(contrastive_dataset)
        else:
            tokenized = self.tokenizer(
                contrastive_dataset["text"],
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            return datasets.Dataset.from_dict(
                {
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": contrastive_dataset["labels"],
                }
            ).with_format("torch")

    def _preprocess_head_dataset(self, head_dataset, eval=False):
        tokenized = self.tokenizer(
            head_dataset["text"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        if not eval:
            # don't drop classes in eval so metrics are accurate
            head_dataset = head_dataset.map(self._maybe_drop_classes, batched=True)
        return datasets.Dataset.from_dict(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": head_dataset["labels"],
            }
        ).with_format("torch")

    def preprocess_predict_dataset(self, predict_dataset):
        tokenized = self.tokenizer(
            predict_dataset["text"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return datasets.Dataset.from_dict(
            {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
        ).with_format("torch")

    def _maybe_drop_classes(self, row):
        # will set the labels of the drop classes to 0, so they will never be predicted
        # this keeps the metrics accurate and the model from learning to predict the drop classes
        if len(self.drop_classes) > 0:
            # we work with lists
            labels = np.asarray(row["labels"])
            indices = [LABEL2INT[drop_class] for drop_class in self.drop_classes]
            labels[:, indices] = 0
            row["labels"] = labels.tolist()
        return row

    def _load_languages_datasets(self, languages, train=True):
        return self._filter_languages(self._get_or_load_datasets(train), languages)

    @staticmethod
    def _filter_languages(dataset, languages):
        return dataset.filter(lambda x: x["language"] in languages)

    @property
    def num_classes(self):
        return len(INT2LABEL)

    def _load_all_datasets(self):
        if self._split_datasets is not None:
            return
        train_languages = set(
            self.languages_for_contrastive + self.languages_for_head_train
        )
        train_only_languages = [
            language
            for language in train_languages
            if language not in self.languages_for_head_eval
        ]
        eval_only_languages = [
            language
            for language in self.languages_for_head_eval
            if language not in train_languages
        ]
        train_and_eval_languages = [
            language
            for language in train_languages
            if language in self.languages_for_head_eval
        ]

        train_only_datasets = [
            self._get_single_named_dataset(language)
            for language in train_only_languages
        ]
        eval_only_datasets = [
            self._get_single_named_dataset(language) for language in eval_only_languages
        ]
        train_and_eval_datasets = [
            self._get_single_named_dataset(language)
            for language in train_and_eval_languages
        ]

        if self.use_dev:
            train_only_datasets += [
                self._get_single_named_dataset(language, dev=True)
                for language in train_only_languages
            ]
            eval_only_datasets += [
                self._get_single_named_dataset(language, dev=True)
                for language in eval_only_languages
            ]
            dev_train_and_eval_datasets = [
                self._get_single_named_dataset(language, dev=True)
                for language in train_and_eval_languages
            ]

            split_train_dataset = concatenate_datasets(
                train_only_datasets + train_and_eval_datasets
            )
            if not (eval_only_datasets or dev_train_and_eval_datasets):
                split_eval_dataset = datasets.Dataset.from_dict({'text': [], 'labels': []}).with_format('torch')
            else:
                split_eval_dataset = concatenate_datasets(
                    eval_only_datasets + dev_train_and_eval_datasets
                )
            train_and_eval_datasets += dev_train_and_eval_datasets
        elif len(train_and_eval_datasets) > 0:
            train_and_eval_dataset = concatenate_datasets(train_and_eval_datasets)

            #  stratify = train_and_eval_dataset.add_column(
                #  "stratify",
                #  self._create_stratify_column(train_and_eval_dataset["language"]),
            #  ).class_encode_column("stratify")
            #  split = stratify.train_test_split(
                #  0.2, seed=42, stratify_by_column="stratify"
            #  )
            #  split.remove_columns("stratify")
            split = train_and_eval_dataset.train_test_split(
                0.2, seed=42
            )

            split_train_dataset = concatenate_datasets(
                train_only_datasets + [split["train"]]
            )
            split_eval_dataset = concatenate_datasets(
                eval_only_datasets + [split["test"]]
            )
        else:  # len(train_and_eval_datasets) == 0
            split_train_dataset = concatenate_datasets(train_only_datasets)
            split_eval_dataset = concatenate_datasets(eval_only_datasets)

        if len(train_only_datasets) > 0:
            self._train_only_dataset = concatenate_datasets(train_only_datasets)
        if len(eval_only_datasets) > 0:
            self._eval_only_dataset = concatenate_datasets(eval_only_datasets)
        if len(train_and_eval_datasets) > 0:
            self._train_and_eval_dataset = concatenate_datasets(train_and_eval_datasets)

        self._split_datasets = (split_train_dataset, split_eval_dataset)

    def _get_or_load_datasets(self, train):
        if self._split_datasets is None:
            self._load_all_datasets()
        return self._split_datasets[0 if train else 1]

    def _create_stratify_column(self, language_list):
        languages = list(set(language_list))
        stratify_column = []
        for language in language_list:
            stratify_column.append(languages.index(language))
        return stratify_column

    def _get_single_named_dataset(self, language, dev=False):
        data_dir = os.path.join(self.data_dir, language)
        prefix = "dev" if dev else "train"
        features_dir_path = f"{data_dir}/{prefix}-articles-subtask-2"
        labels_path = f"{data_dir}/{prefix}-labels-subtask-2.txt"
        labels = pd.read_csv(
            labels_path, sep="\t", header=None, names=["ids", "labels"], index_col="ids"
        )
        features_filenames = [
            os.path.join(features_dir_path, f"article{id}.txt") for id in labels.index
        ]
        features = _get_features_from_files(features_filenames)

        def _attach_labels_and_language(_, idx):
            lbls = labels.iloc[idx]["labels"].split(",")
            return {
                "labels": [
                    float(cat_name in lbls)
                    for cat_name in INT2LABEL
                    # if cat_name not in self.drop_classes
                ],
                "language": language,
            }

        return features.map(_attach_labels_and_language, with_indices=True)

    def _create_contrastive_pairs_dataset(self, dataset):
        return distance_weighted_sentence_pair_generation(
            dataset,
            exponent_base=self.contrastive_weighting_base,
            num_positive_samples=self.num_contrastive_samples,
            num_negative_samples=self.num_contrastive_samples,
        )

    def contrastive_pairs_collate_fn(self, batch):
        tokenizer = self.tokenizer

        num_texts = len(batch[0]["texts"])
        texts = [[] for _ in range(num_texts)]
        labels = []
        categories = []

        for example in batch:
            for idx, text in enumerate(example["texts"]):
                texts[idx].append(text)

            labels.append(example["labels"])
            categories.append(example["category"])

        labels = torch.tensor(labels)
        categories = torch.tensor(categories)

        sentence_features = []
        for text in texts:
            tokenized = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenized.pop("token_type_ids")
            sentence_features.append(tokenized)

        assert len(sentence_features) == 2
        return {
            "input_ids_lhs": sentence_features[0]["input_ids"],
            "attention_mask_lhs": sentence_features[0]["attention_mask"],
            "labels_lhs": labels[:, 0],
            "input_ids_rhs": sentence_features[1]["input_ids"],
            "attention_mask_rhs": sentence_features[1]["attention_mask"],
            "labels_rhs": labels[:, 1],
            "categories": categories,
        }


def _get_features_from_files(features_filenames):
    return load_dataset(
        "text",
        data_files=features_filenames,
        sample_by="document",
        split="train",
    )


@torch.no_grad()
def distance_weighted_sentence_pair_generation(
    dataset,
    exponent_base=np.e,
    num_positive_samples=1,
    num_negative_samples=1,
    metric="cosine",
):
    sentences = dataset["text"]
    labels = torch.tensor(dataset["labels"])
    pairs = []

    distance_matrix = torch.tensor(pairwise_distances(labels, metric=metric, n_jobs=-1))
    # normalize
    distance_matrix = distance_matrix / torch.max(distance_matrix)
    for label_col in range(len(labels[0])):
        idxs_with_positive = labels[:, label_col].argwhere()[:, 0]
        pos_label_distance_matrix = torch.index_select(
            distance_matrix, 0, idxs_with_positive
        )
        pos_label_distance_matrix = torch.index_select(
            pos_label_distance_matrix, 1, idxs_with_positive
        )

        for first_idx in labels[:, label_col].argwhere():
            first_idx = first_idx.item()
            current_sentence = sentences[first_idx]

            num_current_positive_samples = min(
                num_positive_samples, len(idxs_with_positive) - 1
            )
            positive_idxs, positive_similarities = _weighted_pair_sample_selection(
                (idxs_with_positive == first_idx).argwhere().item(),
                pos_label_distance_matrix,
                False,
                num_current_positive_samples,
                exponent_base,
            )
            for positive_idx, _ in zip(positive_idxs, positive_similarities):
                positive_idx = idxs_with_positive[positive_idx]
                positive_sentence = sentences[positive_idx]
                pairs.append(
                    {
                        "texts": [current_sentence, positive_sentence],
                        "labels": [labels[first_idx].tolist(), labels[positive_idx].tolist()],
                        "category": label_col,
                    }
                )

            idxs_with_negative = (
                (labels[:, label_col] == 0).logical_or(
                    torch.arange(len(labels)) == first_idx
                )
            ).argwhere()[:, 0]
            neg_label_distance_matrix = torch.index_select(
                distance_matrix, 0, idxs_with_negative
            )
            neg_label_distance_matrix = torch.index_select(
                neg_label_distance_matrix, 1, idxs_with_negative
            )

            num_current_negative_samples = min(
                num_negative_samples, len(idxs_with_negative) - 1
            )
            negative_idxs, negative_similarities = _weighted_pair_sample_selection(
                (idxs_with_negative == first_idx).argwhere().item(),
                neg_label_distance_matrix,
                True,
                num_current_negative_samples,
                exponent_base,
            )
            for negative_idx, _ in zip(negative_idxs, negative_similarities):
                negative_idx = idxs_with_negative[negative_idx]
                negative_sentence = sentences[negative_idx]
                pairs.append(
                    {
                        "texts": [current_sentence, negative_sentence],
                        "labels": [labels[first_idx].tolist(), labels[negative_idx].tolist()],
                        "category": label_col,
                    }
                )
    return datasets.Dataset.from_pandas(pd.DataFrame(pairs))


def _weighted_pair_sample_selection(
    sample_idx: int,
    normalized_distance_matrix: torch.Tensor,
    bias_similar: bool,
    num_samples: int,
    exponent_base: float,
) -> Tuple[int, float]:
    weight = normalized_distance_matrix[sample_idx, :]
    if bias_similar:
        # low distance -> high probability of choosing
        weight = 1.0 - weight
    # lower distances should be much more likely
    weight = torch.pow(exponent_base, weight)
    # don't choose the sample itself
    weight[sample_idx] = 0
    # to probabilites (normalize), numpy conversion required as otherwise sometimes
    # the probabilities do not sum to 1 for numpy when converting from torch
    weight = np.asarray(weight)
    weight /= np.sum(weight)

    index = np.random.choice(
        np.arange(len(weight)), num_samples, p=weight, replace=False
    )
    sample_similarity = 1.0 - normalized_distance_matrix[sample_idx, index]
    return index, sample_similarity


def _vec_to_label_names(vec):
    names = []
    for i, x in enumerate(vec):
        if x:
            names.append(INT2LABEL[i])
    return ','.join(names)
