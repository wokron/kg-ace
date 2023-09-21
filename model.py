from flair.models import TextClassifier
from flair.data import Sentence
from flair.embeddings import DocumentEmbeddings
import torch


class KGClassifier(TextClassifier):
    def __init__(
        self, embeddings: DocumentEmbeddings, label_type: str, **classifierargs
    ):
        self.selection = None  # for embedding selection
        super().__init__(embeddings, label_type, **classifierargs)

    def _get_embedding_for_data_point(
        self, prediction_data_point: Sentence
    ) -> torch.Tensor:
        embedding_names = self.embeddings.get_names()
        embeddings = prediction_data_point.get_each_embedding(embedding_names)
        return torch.cat(embeddings, dim=0)  # this is prepared for selection
