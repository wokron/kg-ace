import argparse
import logging
from pathlib import Path
from typing import Optional
import yaml
import os

from flair.data import Corpus, Tokenizer
from flair.datasets import CSVClassificationCorpus
from flair.samplers import ImbalancedClassificationDatasetSampler
import flair.embeddings
import embeddings
from flair.nn import Model
from model import KGClassifier, EmbedController
from flair.trainers import ModelTrainer
from tokenizer import FlairBertTokenizer
import torch

log = logging.getLogger("flair")


# 保存加载
class TrainState:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        if os.path.exists(save_dir / "train_state.pt"):
            self.load()
        else:
            save_dir.mkdir(exist_ok=True, parents=True)
            self.episode = 0
            self.action_dict = {}
            self.best_action = None
            self.baseline_score = 0
            self.best_episode = -1
            self.agent_dict = None
            self.save()

    def load(self):
        state_dict = torch.load(self.save_dir / "train_state.pt")
        self.episode = state_dict["episode"]
        self.action_dict = state_dict["action_dict"]
        self.best_action = state_dict["best_action"]
        self.baseline_score = state_dict["baseline_score"]
        self.best_episode = state_dict["best_episode"]
        self.agent_dict = state_dict["agent_dict"]

    def save(self):
        state_dict = {
            "episode": self.episode,
            "action_dict": self.action_dict,
            "best_action": self.best_action,
            "baseline_score": self.baseline_score,
            "best_episode": self.best_episode,
            "agent_dict": self.agent_dict,
        }
        torch.save(state_dict, self.save_dir / "train_state.pt")


# 创建一个分类任务的语料库
def get_corpus(data_folder, data_sep, tokenizer: Tokenizer):
    column_name_map = {1: "text", 2: "label"}
    corpus: Corpus = CSVClassificationCorpus(
        data_folder,
        column_name_map,
        skip_header=True,
        label_type="label",
        delimiter=data_sep,
        tokenizer=tokenizer,
    )
    return corpus


# 根据提供的配置信息，创建并返回一组标记嵌入对象的列表
def create_token_embeddings_list(token_embeddings_config: dict):
    token_embeddings_list = []
    for embed_name in token_embeddings_config.keys():
        config = token_embeddings_config[embed_name]
        embed_class_name = embed_name.split("-")[0]
        if hasattr(flair.embeddings, embed_class_name):
            embedding = getattr(flair.embeddings, embed_class_name)(**config)
            token_embeddings_list.append(embedding)
        else:
            raise Exception()
    return token_embeddings_list


def create_embeddings(embedding_config: dict):
    document_embedding_config = embedding_config["document_embedding"]
    token_embeddings_config = embedding_config.get("token_embeddings", {})

    token_embeddings_list = create_token_embeddings_list(token_embeddings_config)

    document_embedding_class = document_embedding_config["type"]
    if hasattr(embeddings, "My" + document_embedding_class):
        embedding = getattr(embeddings, "My" + document_embedding_class)(
            embeddings=token_embeddings_list, **document_embedding_config["config"]
        )
        return embedding
    elif hasattr(flair.embeddings, document_embedding_class):
        if "Transformer" in document_embedding_class:
            embedding = getattr(flair.embeddings, document_embedding_class)(
                **document_embedding_config["config"]
            )
        else:
            embedding = getattr(flair.embeddings, document_embedding_class)(
                embeddings=token_embeddings_list, **document_embedding_config["config"]
            )
        return embedding
    else:
        return None, None


# 据提供的配置信息和输入参数创建一个 KGClassifier 模型对象，并将其返回。
def create_model(model_config, embeddings, label_type, label_dict):
    return KGClassifier(
        embeddings, label_type, label_dictionary=label_dict, **model_config
    )


# 这个函数的作用是允许用户在原有模型训练的基础上进行继续训练，并且可以通过传入额外的参数来改变或增加训练参数，例如增加额外的训练轮数等。
def my_resume(
    trainer: ModelTrainer,
    model: Model,
    additional_epochs: Optional[int] = None,
    **trainer_args,
):
    assert model.model_card is not None
    trainer.model = model
    # recover all arguments that were used to train this model
    args_used_to_train_model = model.model_card["training_parameters"]

    # you can overwrite params with your own
    for param in trainer_args:
        args_used_to_train_model[param] = trainer_args[param]
        if param == "optimizer" and "optimizer_state_dict" in args_used_to_train_model:
            del args_used_to_train_model["optimizer_state_dict"]
        if param == "scheduler" and "scheduler_state_dict" in args_used_to_train_model:
            del args_used_to_train_model["scheduler_state_dict"]

    # surface nested arguments
    kwargs = args_used_to_train_model["kwargs"]
    del args_used_to_train_model["kwargs"]

    if additional_epochs is not None:
        args_used_to_train_model["max_epochs"] = (
            args_used_to_train_model.get("epoch", kwargs.get("epoch", 0))
            + additional_epochs
        )

    # resume training with these parameters
    return trainer.train(**args_used_to_train_model, **kwargs)


# 主函数
def main(args, config: dict):
    output_dir = Path(args.output_dir)
    embeddings_config = config["embedding"]  # embedding is must
    model_config = config.get("model", {})  # model is optional
    train_config = config.get("train", {})  # train is optional
    label_type = "label"

    tokenizer = FlairBertTokenizer()

    corpus = get_corpus(args.data_folder, args.data_sep, tokenizer)
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    embeddings = create_embeddings(embeddings_config)

    model_dir = output_dir / args.name

    train_state = TrainState(model_dir)

    cur_episode = train_state.episode

    embed_agent = EmbedController(
        num_actions=len(embeddings_config["token_embeddings"]), mode=args.select_mode
    )
    if train_state.agent_dict is not None:
        embed_agent.load_state_dict(train_state.agent_dict)

    for episode in range(cur_episode, args.max_episodes):
        log.info(f"--- start episode {episode} ---")
        action, log_prob = embed_agent.sample(episode == 0)
        log.info(f"selection: {action}")

        model_base_path = model_dir / f"{args.name}-{episode:05d}"
        if os.path.exists(model_base_path / "checkpoint.pt"):
            log.info("continue epoch training")
            model = KGClassifier.load(model_base_path / "checkpoint.pt")
            model.embeddings.selection = action
            trainer = ModelTrainer(model, corpus)
            result = my_resume(
                trainer,
                model,
                base_path=model_base_path,
            )
        elif episode == 0:
            log.info("start training a new model")
            model = create_model(model_config, embeddings, label_type, label_dict)
            model.embeddings.selection = action
            trainer = ModelTrainer(model, corpus)
            result = trainer.train(
                model_base_path,
                checkpoint=True,
                sampler=ImbalancedClassificationDatasetSampler,
                **train_config,
            )
        else:
            log.info("continue training")
            prev_trained_model_path = (
                model_dir / f"{args.name}-{episode-1:05d}" / "checkpoint.pt"
            )
            model = KGClassifier.load(prev_trained_model_path)
            model.embeddings.selection = action
            trainer = ModelTrainer(model, corpus)
            result = my_resume(
                trainer,
                model,
                train_config.get("max_epochs", 10),
                base_path=model_base_path,
            )

        log.info(f"result: {result}")
        log.info(f"--- finish training of episode {episode} ---")

        score = max(result["dev_score_history"] + [0.0])
        embed_agent.learn(
            score,
            train_state.action_dict,
            action,
            log_prob,
            first_episode=(episode == 0),
        )
        action_key = tuple(action.cpu().tolist())
        if action_key not in train_state.action_dict.keys():
            train_state.action_dict[action_key] = {
                "counts": 0,
                "scores": [],
            }
        train_state.action_dict[action_key]["counts"] += 1
        train_state.action_dict[action_key]["scores"].append(score)
        train_state.episode += 1
        train_state.agent_dict = embed_agent.state_dict()

        train_state.save()

        log.info(f"--- end episode {episode} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_folder", default=".")
    parser.add_argument("--data_sep", default="\t")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--max_episodes", type=int, default=10)
    parser.add_argument("--name", default="kg-ace")
    parser.add_argument("--select_mode", type=str, default="DEFAULT")

    args = parser.parse_args()

    with open(
        args.config, "r", encoding="UTF-8"
    ) as config_file:  # 加了, encoding='UTF-8'
        config = yaml.safe_load(config_file)

    log.info("config: ", config)

    main(args, config)
