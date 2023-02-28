import torch
from torch import nn
import torch.nn.functional as F


class WeightedCosineSimilarityLoss(nn.Module):
    """Docstring for WeightedCosineSimilarityLoss."""

    def __init__(self, class_num: int):
        super(WeightedCosineSimilarityLoss, self).__init__()
        self.class_num = class_num

    def forward(self, features, labels=None):
        device = features.device
        temp_labels = labels
        loss = torch.tensor(0, device=device, dtype=torch.float64)
        for i in range(self.class_num):
            pos_idx = torch.where(temp_labels[:, i] == 1)[0]
            if len(pos_idx) == 0:
                continue
            neg_idx = torch.where(temp_labels[:, i] != 1)[0]
            pos_samples = features[pos_idx, :]
            neg_samples = features[neg_idx, :]
            size = neg_samples.shape[0] + 1
            dist = self.hamming_distance_by_matrix(temp_labels)
            pos_weight = 1 - dist[pos_idx, :][:, pos_idx] / self.class_num
            neg_weight = dist[pos_idx, :][:, neg_idx]
            pos_dis = self.exp_cosine_sim(pos_samples, pos_samples) * pos_weight
            neg_dis = self.exp_cosine_sim(pos_samples, neg_samples) * neg_weight
            denominator = neg_dis.sum(1) + pos_dis
            loss += torch.mean(torch.log(denominator / (pos_dis * size)))
        return loss

    def hamming_distance_by_matrix(self, labels):
        return torch.matmul(labels, (1 - labels).T) + torch.matmul(1 - labels, labels.T)

    def exp_cosine_sim(self, x1, x2, eps=1e-15, temperature=1):
        w1 = x1.norm(p=2, dim=1, keepdim=True)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        return torch.exp(
            torch.matmul(x1, x2.t()) / ((w1 * w2.t()).clamp(min=eps) * temperature)
        )


class ContrastLoss2(nn.Module):
    """Docstring for ContrastLoss2."""

    def __init__(self, temp=10):
        super(ContrastLoss2, self).__init__()
        self.temp = temp

    def forward(self, features, labels=None):
        C = torch.matmul(labels, labels.T).fill_diagonal_(0)
        beta = C / C.sum(1)
        D = torch.exp(-torch.cdist(features, features) / self.temp)
        D2 = D.sum(1) - 1
        losses = (-beta * torch.log(D / D2))

        return torch.sum(losses)


class MultiCategoricalSiameseLoss(nn.Module):
    """Docstring for MultiCategoricalSiameseLoss."""

    def __init__(
        self, sentence_embedding_dimension: int, num_classes: int, simple=False
    ):
        super(MultiCategoricalSiameseLoss, self).__init__()
        self.multi_cat_linear = nn.Sequential(
            nn.Linear(num_classes, sentence_embedding_dimension),
            nn.ReLU(),
        )
        self.output_similarity = nn.Sequential(
            nn.Dropout(),
            nn.Linear(sentence_embedding_dimension, 1),
        )

        # simple
        self.simple = simple
        self.simple_nn = nn.Sequential(
            nn.Dropout(),
            nn.Linear(sentence_embedding_dimension, num_classes),
        )
        self.num_classes = num_classes
        self.loss = nn.BCEWithLogitsLoss()

    def forward(
        self, embedding_lhs, embedding_rhs, categories: torch.Tensor, labels
    ) -> torch.Tensor:
        embeddings_diff = torch.abs(embedding_lhs - embedding_rhs)
        if self.simple:
            return self.loss(self.simple_nn(embeddings_diff)[:, categories], labels)

        category_embedding_attention = self.multi_cat_linear(
            F.one_hot(categories, num_classes=self.num_classes).to(torch.float)
        )
        return self.loss(
            self.output_similarity(
                torch.mul(embeddings_diff, category_embedding_attention)
            ).squeeze(-1),
            labels,
        )

    def compute_train_pairs(self, labels):
        train_pairs = []
        pos_similarity = torch.tensor([1], dtype=torch.float64, device=labels.device)
        neg_similarity = torch.tensor([0], dtype=torch.float64, device=labels.device)
        for class_idx in range(self.num_classes):
            pos_indices = labels[:, class_idx].nonzero().squeeze(-1)
            if len(pos_indices) == 0:
                continue
            neg_indices = (labels[:, class_idx] == 0).nonzero().squeeze(-1)

            for i, first_idx in enumerate(pos_indices):
                for pos_idx in pos_indices[i + 1 :]:
                    train_pairs.append(
                        {
                            "indices": (first_idx, pos_idx),
                            "class_idx": class_idx,
                            "similarity": pos_similarity,
                        }
                    )

                for neg_idx in neg_indices:
                    train_pairs.append(
                        {
                            "indices": (first_idx, neg_idx),
                            "class_idx": class_idx,
                            "similarity": neg_similarity,
                        }
                    )
        return train_pairs

    def forward_batch(self, sentence_embeddings, labels):
        losses = []
        for batch in self.compute_train_pairs(labels):
            first_idx, second_idx, class_idx, similarity = (
                batch["indices"][0],
                batch["indices"][1],
                batch["class_idx"],
                batch["similarity"],
            )
            category = (
                torch.ones(1, dtype=torch.int64).to(sentence_embeddings.device)
                * class_idx
            )
            category_embedding_attention = self.multi_cat_linear(
                F.one_hot(
                    category,
                    num_classes=self.num_classes,
                ).to(torch.float)
            )
            embeddings_diff = torch.abs(
                sentence_embeddings[first_idx] - sentence_embeddings[second_idx]
            )
            output_similarity = self.output_similarity(
                torch.mul(embeddings_diff, category_embedding_attention)
            ).squeeze(-1)
            losses.append(self.loss(output_similarity, similarity))
        return sum(losses)
