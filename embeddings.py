import logging
from typing import Any, Dict, List, Optional, Union, cast

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn import RNNBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import flair
from flair.data import Sentence
from flair.embeddings.base import (
    DocumentEmbeddings,
    load_embeddings,
    register_embeddings,
)
from flair.embeddings.token import FlairEmbeddings, StackedEmbeddings, TokenEmbeddings
from flair.embeddings.transformer import (
    TransformerEmbeddings,
    TransformerOnnxDocumentEmbeddings,
)
from flair.nn import LockedDropout, WordDropout

log = logging.getLogger("flair")


@register_embeddings
class MyDocumentPoolEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        embeddings: Union[TokenEmbeddings, List[TokenEmbeddings]],
        fine_tune_mode: str = "none",
        pooling: str = "mean",
        selection = None,
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param fine_tune_mode: if set to "linear" a trainable layer is added, if set to
        "nonlinear", a nonlinearity is added as well. Set this to make the pooling trainable.
        :param pooling: a string which can any value from ['mean', 'max', 'min']
        """
        super().__init__()

        self.selection = selection  # this is important

        if isinstance(embeddings, StackedEmbeddings):
            embeddings = embeddings.embeddings
        elif isinstance(embeddings, TokenEmbeddings):
            embeddings = [embeddings]

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)
        self.__embedding_length = self.embeddings.embedding_length

        # optional fine-tuning on top of embedding layer
        self.fine_tune_mode = fine_tune_mode
        if self.fine_tune_mode in ["nonlinear", "linear"]:
            self.embedding_flex = torch.nn.Linear(self.embedding_length, self.embedding_length, bias=False)
            self.embedding_flex.weight.data.copy_(torch.eye(self.embedding_length))

        if self.fine_tune_mode in ["nonlinear"]:
            self.embedding_flex_nonlinear = torch.nn.ReLU()
            self.embedding_flex_nonlinear_map = torch.nn.Linear(self.embedding_length, self.embedding_length)

        self.__embedding_length = self.embeddings.embedding_length

        self.to(flair.device)

        if pooling not in ["min", "max", "mean"]:
            raise ValueError(f"Pooling operation for {self.mode!r} is not defined")

        self.pooling = pooling
        self.name: str = f"document_{self.pooling}"

        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def embed(self, sentences: Union[List[Sentence], Sentence]):
        """Add embeddings to every sentence in the given list of sentences. If embeddings are already added, updates
        only if embeddings are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if isinstance(sentences, Sentence):
            sentences = [sentences]

        self.embeddings.embed(sentences)

        for sentence in sentences:
            if self.selection is not None:
                word_embeddings = []
                for token in sentence.tokens:
                    masked_embeddings = []
                    for idx, embedding in enumerate(token.get_each_embedding()):
                        masked_embeddings.append(embedding * self.selection[idx])
                    masked_embeddings = torch.cat(masked_embeddings, dim=0)
                    word_embeddings.append(masked_embeddings.unsqueeze(0))
                word_embeddings = torch.cat(word_embeddings, dim=0).to(flair.device)
            else:
                word_embeddings = torch.cat([token.get_embedding().unsqueeze(0) for token in sentence.tokens], dim=0).to(
                    flair.device
                )

            if self.fine_tune_mode in ["nonlinear", "linear"]:
                word_embeddings = self.embedding_flex(word_embeddings)

            if self.fine_tune_mode in ["nonlinear"]:
                word_embeddings = self.embedding_flex_nonlinear(word_embeddings)
                word_embeddings = self.embedding_flex_nonlinear_map(word_embeddings)

            if self.pooling == "mean":
                pooled_embedding = torch.mean(word_embeddings, 0)
            elif self.pooling == "max":
                pooled_embedding, _ = torch.max(word_embeddings, 0)
            elif self.pooling == "min":
                pooled_embedding, _ = torch.min(word_embeddings, 0)

            sentence.set_embedding(self.name, pooled_embedding)

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        pass

    def extra_repr(self):
        return f"fine_tune_mode={self.fine_tune_mode}, pooling={self.pooling}"

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "MyDocumentPoolEmbeddings":
        embeddings = cast(StackedEmbeddings, load_embeddings(params.pop("embeddings"))).embeddings
        selection = params.pop("selection")
        return cls(embeddings=embeddings, selection=selection, **params)

    def to_params(self) -> Dict[str, Any]:
        return {
            "pooling": self.pooling,
            "fine_tune_mode": self.fine_tune_mode,
            "embeddings": self.embeddings.save_embeddings(False),
            "selection": self.selection
        }
    

@register_embeddings
class MyDocumentRNNEmbeddings(DocumentEmbeddings):
    def __init__(
        self,
        embeddings: List[TokenEmbeddings],
        hidden_size=128,
        rnn_layers=1,
        reproject_words: bool = True,
        reproject_words_dimension: int = None,
        bidirectional: bool = False,
        dropout: float = 0.5,
        word_dropout: float = 0.0,
        locked_dropout: float = 0.0,
        rnn_type="GRU",
        fine_tune: bool = True,
        selection = None,
    ):
        """The constructor takes a list of embeddings to be combined.
        :param embeddings: a list of token embeddings
        :param hidden_size: the number of hidden states in the rnn
        :param rnn_layers: the number of layers for the rnn
        :param reproject_words: boolean value, indicating whether to reproject the token embeddings in a separate linear
        layer before putting them into the rnn or not
        :param reproject_words_dimension: output dimension of reprojecting token embeddings. If None the same output
        dimension as before will be taken.
        :param bidirectional: boolean value, indicating whether to use a bidirectional rnn or not
        :param dropout: the dropout value to be used
        :param word_dropout: the word dropout value to be used, if 0.0 word dropout is not used
        :param locked_dropout: the locked dropout value to be used, if 0.0 locked dropout is not used
        :param rnn_type: 'GRU' or 'LSTM'
        """
        super().__init__()

        self.selection = selection

        self.embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embeddings)

        self.rnn_type = rnn_type

        self.reproject_words = reproject_words
        self.bidirectional = bidirectional

        self.length_of_all_token_embeddings: int = self.embeddings.embedding_length

        self.static_embeddings = False if fine_tune else True

        self.__embedding_length: int = hidden_size
        if self.bidirectional:
            self.__embedding_length *= 4

        self.embeddings_dimension: int = self.length_of_all_token_embeddings
        if self.reproject_words and reproject_words_dimension is not None:
            self.embeddings_dimension = reproject_words_dimension

        self.word_reprojection_map = torch.nn.Linear(self.length_of_all_token_embeddings, self.embeddings_dimension)

        # bidirectional RNN on top of embedding layer
        if rnn_type == "LSTM":
            self.rnn: RNNBase = torch.nn.LSTM(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = torch.nn.GRU(
                self.embeddings_dimension,
                hidden_size,
                num_layers=rnn_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
            )

        self.name = "document_" + self.rnn._get_name()

        # dropouts
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else None
        self.locked_dropout = LockedDropout(locked_dropout) if locked_dropout > 0.0 else None
        self.word_dropout = WordDropout(word_dropout) if word_dropout > 0.0 else None

        torch.nn.init.xavier_uniform_(self.word_reprojection_map.weight)

        self.to(flair.device)

        self.eval()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]):
        """Add embeddings to all sentences in the given list of sentences. If embeddings are already added, update
        only if embeddings are non-static."""

        # TODO: remove in future versions
        if not hasattr(self, "locked_dropout"):
            self.locked_dropout = None
        if not hasattr(self, "word_dropout"):
            self.word_dropout = None

        self.rnn.zero_grad()
        # embed words in the sentence
        self.embeddings.embed(sentences)

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs: List[torch.Tensor] = list()
        for sentence in sentences:
            if self.selection is not None:
                all_embs += [emb * self.selection[idx] for token in sentence for idx, emb in enumerate(token.get_each_embedding())]
            else:
                all_embs += [emb for token in sentence for emb in token.get_each_embedding()]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[: self.embeddings.embedding_length * nb_padding_tokens]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        # before-RNN dropout
        if self.dropout:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.locked_dropout:
            sentence_tensor = self.locked_dropout(sentence_tensor)
        if self.word_dropout:
            sentence_tensor = self.word_dropout(sentence_tensor)

        # reproject if set
        if self.reproject_words:
            sentence_tensor = self.word_reprojection_map(sentence_tensor)

        # push through RNN
        packed = pack_padded_sequence(sentence_tensor, lengths, enforce_sorted=False, batch_first=True)  # type: ignore
        rnn_out, hidden = self.rnn(packed)
        outputs, output_lengths = pad_packed_sequence(rnn_out, batch_first=True)

        # after-RNN dropout
        if self.dropout:
            outputs = self.dropout(outputs)
        if self.locked_dropout:
            outputs = self.locked_dropout(outputs)

        # extract embeddings from RNN
        for sentence_no, length in enumerate(lengths):
            last_rep = outputs[sentence_no, length - 1]

            embedding = last_rep
            if self.bidirectional:
                first_rep = outputs[sentence_no, 0]
                embedding = torch.cat([first_rep, last_rep], 0)

            if self.static_embeddings:
                embedding = embedding.detach()

            sentence = sentences[sentence_no]
            sentence.set_embedding(self.name, embedding)

    def _apply(self, fn):
        # models that were serialized using torch versions older than 1.4.0 lack the _flat_weights_names attribute
        # check if this is the case and if so, set it
        for child_module in self.children():
            if isinstance(child_module, torch.nn.RNNBase) and not hasattr(child_module, "_flat_weights_names"):
                _flat_weights_names = []

                if child_module.__dict__["bidirectional"]:
                    num_direction = 2
                else:
                    num_direction = 1
                for layer in range(child_module.__dict__["num_layers"]):
                    for direction in range(num_direction):
                        suffix = "_reverse" if direction == 1 else ""
                        param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                        if child_module.__dict__["bias"]:
                            param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]
                        param_names = [x.format(layer, suffix) for x in param_names]
                        _flat_weights_names.extend(param_names)

                setattr(child_module, "_flat_weights_names", _flat_weights_names)

            child_module._apply(fn)

    def to_params(self):
        # serialize the language models and the constructor arguments (but nothing else)
        model_state = {
            "embeddings": self.embeddings.save_embeddings(False),
            "hidden_size": self.rnn.hidden_size,
            "rnn_layers": self.rnn.num_layers,
            "reproject_words": self.reproject_words,
            "reproject_words_dimension": self.embeddings_dimension,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout.p if self.dropout is not None else 0.0,
            "word_dropout": self.word_dropout.p if self.word_dropout is not None else 0.0,
            "locked_dropout": self.locked_dropout.p if self.locked_dropout is not None else 0.0,
            "rnn_type": self.rnn_type,
            "fine_tune": not self.static_embeddings,
            "selection": self.selection,
        }

        return model_state

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "MyDocumentRNNEmbeddings":
        stacked_embeddings = load_embeddings(params["embeddings"])
        assert isinstance(stacked_embeddings, StackedEmbeddings)
        return cls(
            embeddings=stacked_embeddings.embeddings,
            hidden_size=params["hidden_size"],
            rnn_layers=params["rnn_layers"],
            reproject_words=params["reproject_words"],
            reproject_words_dimension=params["reproject_words_dimension"],
            bidirectional=params["bidirectional"],
            dropout=params["dropout"],
            word_dropout=params["word_dropout"],
            locked_dropout=params["locked_dropout"],
            rnn_type=params["rnn_type"],
            fine_tune=params["fine_tune"],
            selection=params["selection"],
        )

    def __setstate__(self, d):
        # re-initialize language model with constructor arguments
        language_model = MyDocumentRNNEmbeddings(
            embeddings=d["embeddings"],
            hidden_size=d["hidden_size"],
            rnn_layers=d["rnn_layers"],
            reproject_words=d["reproject_words"],
            reproject_words_dimension=d["reproject_words_dimension"],
            bidirectional=d["bidirectional"],
            dropout=d["dropout"],
            word_dropout=d["word_dropout"],
            locked_dropout=d["locked_dropout"],
            rnn_type=d["rnn_type"],
            fine_tune=d["fine_tune"],
            selection=d["selection"],
        )

        # special handling for deserializing language models
        if "state_dict" in d:
            language_model.load_state_dict(d["state_dict"])

        # copy over state dictionary to self
        for key in language_model.__dict__.keys():
            self.__dict__[key] = language_model.__dict__[key]

        # set the language model to eval() by default (this is necessary since FlairEmbeddings "protect" the LM
        # in their "self.train()" method)
        self.eval()