from flair.models import TextClassifier
from flair.data import Sentence
from flair.embeddings import DocumentEmbeddings, TokenEmbeddings, StackedEmbeddings
import torch
from torch import nn
import flair
from typing import Literal, Union, List
from flair.embeddings.base import register_embeddings


class KGClassifier(TextClassifier):
    def __init__(
        self,
        embeddings: DocumentEmbeddings,
        label_type: str,
        selection: torch.Tensor = torch.Tensor([1, 2, 5, 6, 7, 8, 9]),
        **classifierargs
    ):
        self.selection: torch.Tensor = selection  # for embedding selection
        super().__init__(embeddings, label_type, **classifierargs)

    def _get_embedding_for_data_point(
        self, prediction_data_point: Sentence
    ) -> torch.Tensor:
        embedding_names = self.embeddings.get_names()
        embeddings = prediction_data_point.get_each_embedding(embedding_names)
        return torch.cat(embeddings, dim=0)  # this is prepared for selection

    def _get_state_dict(self):
        model_state = {
            **super()._get_state_dict(),
            "selection": self.selection,
        }
        return model_state

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        return super()._init_model_with_state_dict(
            state,
            selection=state.get("selection"),
            **kwargs,
        )


class EmbedController(torch.nn.Module):
    def __init__(
        self,
        num_actions: int,
        discount: float = 0.5,
        optimizer_class=torch.optim.Adam,
        mode: Literal["DEFAULT", "ALL", "RANDOM"] = "DEFAULT",
        optimizer_kwargs: dict = {"lr": 0.1},
    ):
        super(EmbedController, self).__init__()
        self.num_actions = num_actions
        self.discount = discount
        self.selector = nn.Parameter(torch.zeros(self.num_actions), requires_grad=True)
        self.optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)
        self.previous_action = None
        self.best_action = None
        self.mode = mode

        self.to(flair.device)

    def sample(self, first_episode=False):
        if self.mode == "DEFAULT":
            return self.__default_sample(first_episode)
        elif self.mode == "ALL":
            return self.__all_one_sample()
        elif self.mode == "RANDOM":
            return self.__random_sample()
        else:
            raise Exception("invalid sample mode")

    def __default_sample(self, first_episode=False):
        if first_episode:
            return self.__all_one_sample()
        else:
            return self.__random_sample()

    def __random_sample(self):
        value = self.get_value()
        one_prob = torch.sigmoid(value)
        m = torch.distributions.Bernoulli(one_prob)
        # avoid all values are 0, or avoid the selection is the same as previous iteration in training
        action = m.sample()
        while action.sum() == 0 or (
            self.previous_action is not None and (self.previous_action == action).all()
        ):
            action = m.sample()

        log_prob = m.log_prob(action)
        self.previous_action = action.clone()
        return action, log_prob

    def __all_one_sample(self):
        log_prob = torch.log(torch.sigmoid(self.get_value()))
        action = torch.ones(self.num_actions)
        return action, log_prob

    def learn(
        self,
        best_score,
        prev_action_dict: dict,
        cur_action: torch.Tensor,
        log_prob: torch.Tensor,
        first_episode: bool = False,
    ):
        if self.mode != "DEFAULT":
            return

        if first_episode:
            self.best_action = cur_action
        else:
            self.optimizer.zero_grad()
            self.zero_grad()
            controller_loss = 0
            reward_at_each_position = torch.zeros_like(cur_action)
            for prev_action in prev_action_dict.keys():
                reward = best_score - max(prev_action_dict[prev_action]["scores"])
                prev_action = torch.tensor(prev_action).type_as(cur_action)
                reward *= self.discount ** (
                    torch.abs(cur_action - prev_action).sum() - 1
                )
                reward_at_each_position += reward * torch.abs(cur_action - prev_action)
            controller_loss -= (log_prob * reward_at_each_position).sum()
            controller_loss.backward()
            self.optimizer.step()

    def forward(self, states=None, mask=None):
        value = self.get_value(states, mask)

        return torch.sigmoid(value)

    def get_value(self):
        value = self.selector
        return value
