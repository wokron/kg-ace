import argparse
from pathlib import Path
import yaml
import os

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus
import flair.embeddings
from model import KGClassifier
from flair.trainers import ModelTrainer
import torch


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


def train(trainer: ModelTrainer, base_path, train_config):
    return trainer.train(base_path, **train_config)


def main(args, config: dict):
    output_dir = Path(args.output_dir)
    label_type = "label"

    corpus = get_corpus(args.data_folder, args.data_sep)
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    embeddings_config = config["embedding"]  # embedding is must
    embeddings = create_embeddings(embeddings_config)

    model_config = config.get("model", {})  # model is optional
    train_config = config.get("train", {})  # train is optional

    model_dir = output_dir / args.name

    if os.path.exists(model_dir / "ckpt.pt"):
        train_state = torch.load(model_dir / "ckpt.pt")
    else:
        train_state = {"episode": 0}

    for episode in range(train_state["episode"], args.max_episodes):
        if episode == 0:
            model = create_model(model_config, embeddings, label_type, label_dict)
        else:
            model = KGClassifier.load(
                model_dir / f"{args.name}-{episode-1:05d}" / "best-model.pt"
            )

        trainer = ModelTrainer(model, corpus)

        model_base_path = model_dir / f"{args.name}-{episode:05d}"
        result = train(trainer, model_base_path, train_config)
        print(result)

        train_state["episode"] += 1
        torch.save(train_state, model_dir / "ckpt.pt")


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

    print("config: ", config)
    print("config: ", type(config))

    main(args, config)
