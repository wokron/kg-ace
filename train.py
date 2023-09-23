import argparse
import logging
from pathlib import Path
import yaml
import os

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
import flair.embeddings
from model import KGClassifier
from flair.trainers import ModelTrainer
import torch

log = logging.getLogger("flair")


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
            self.save()

    def load(self):
        state_dict = torch.load(self.save_dir / "train_state.pt")
        self.episode = state_dict["episode"]
        self.action_dict = state_dict["action_dict"]
        self.best_action = state_dict["best_action"]
        self.baseline_score = state_dict["baseline_score"]
        self.best_episode = state_dict["best_episode"]

    def save(self):
        state_dict = {
            "episode": self.episode,
            "action_dict": self.action_dict,
            "best_action": self.best_action,
            "baseline_score": self.baseline_score,
            "best_episode": self.best_episode,
        }
        torch.save(state_dict, self.save_dir / "train_state.pt")


def get_corpus(data_folder, data_sep):
    column_name_map = {1: "text", 2: "label"}
    corpus: Corpus = CSVClassificationCorpus(
        data_folder,
        column_name_map,
        skip_header=True,
        label_type="label",
        delimiter=data_sep,
    )
    return corpus


def create_token_embeddings_list(token_embeddings_config: dict):
    token_embeddings_list = []
    for embed_name in token_embeddings_config.keys():
        config = token_embeddings_config[embed_name]
        embed_class_name = embed_name.split("-")[0]
        if hasattr(flair.embeddings, embed_class_name):
            embedding = getattr(flair.embeddings, embed_class_name)(**config)
            token_embeddings_list.append(embedding)
    return token_embeddings_list


def create_embeddings(embedding_config: dict):
    document_embedding_config = embedding_config["document_embedding"]
    token_embeddings_config = embedding_config["token_embeddings"]

    token_embeddings_list = create_token_embeddings_list(token_embeddings_config)

    document_embedding_class = document_embedding_config["type"]
    if hasattr(flair.embeddings, document_embedding_class):
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
        return None


def create_model(model_config, embeddings, label_type, label_dict):
    return KGClassifier(
        embeddings, label_type, label_dictionary=label_dict, **model_config
    )


def main(args, config: dict):
    output_dir = Path(args.output_dir)
    embeddings_config = config["embedding"]  # embedding is must
    model_config = config.get("model", {})  # model is optional
    train_config = config.get("train", {})  # train is optional
    label_type = "label"

    corpus = get_corpus(args.data_folder, args.data_sep)
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    embeddings = create_embeddings(embeddings_config)

    model_dir = output_dir / args.name

    train_state = TrainState(model_dir)

    cur_episode = train_state.episode

    for episode in range(cur_episode, args.max_episodes):
        log.info(f"--- start episode {episode} ---")
        model_base_path = model_dir / f"{args.name}-{episode:05d}"
        if episode == 0:
            log.info("start training a new model")
            model = create_model(model_config, embeddings, label_type, label_dict)
            trainer = ModelTrainer(model, corpus)
            result = trainer.fine_tune(model_base_path, checkpoint=True, **train_config)
        else:
            log.info("continue training")
            prev_trained_model_path = model_dir / f"{args.name}-{episode-1:05d}" / "checkpoint.pt"
            model = KGClassifier.load(prev_trained_model_path)
            trainer = ModelTrainer(model, corpus)
            result = trainer.resume(model, train_config.get("max_epochs", 10), base_path=model_base_path)

        log.info(f"result: {result}")

        log.info(f"--- finish training of episode {episode} ---")

        train_state.episode += 1
        train_state.save()

        log.info(f"--- end episode {episode} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_folder", default=".")
    parser.add_argument("--data_sep", default="\t")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--max_episodes", type=int, default=2)
    parser.add_argument("--name", default="kg-ace")

    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    log.info("config: ", config)

    main(args, config)
