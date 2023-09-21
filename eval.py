from flair.data import Sentence
from flair.models import TextClassifier
import heapq
import numpy as np


def get_score(s: Sentence):
    pos_label = list(filter(lambda elm: elm.type == "1", s.labels))[0]
    return pos_label.score()


def get_rank(pos_sentence: Sentence, neg_sentences: list[Sentence]):
    pos_score = get_score(pos_sentence)
    neg_scores = []
    for s in neg_sentences:
        neg_score = get_score(s)
        neg_scores.append(neg_score)
    neg_scores: np.ndarray = np.array(neg_scores)
    return (neg_scores > pos_score).sum() + 1


def hit_at_k(ranks: np.ndarray, k: int):
    return (ranks <= k).mean()


def mean_rank(ranks: np.ndarray):
    return ranks.mean()


def mean_reciprocal_ranking(ranks: np.ndarray):
    return (1 / ranks).mean()
