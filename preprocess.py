from pathlib import Path
import random
import pandas as pd
import argparse
from tqdm import tqdm


def get_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path, sep="\t")
    dataset.columns = ["head", "relation", "tail"]
    dataset["label"] = "1"
    return dataset


def get_entity2text(path):
    data = pd.read_csv(path, sep="\t")
    data.columns = ["entity", "text"]
    return data


def get_relation2text(path):
    data = pd.read_csv(path, sep="\t")
    data.columns = ["relation", "text"]
    return data


def get_entities(path):
    data = pd.read_csv(path, sep="\t")
    data.columns = ["entity"]
    return data


def random_choice(elm_set: list, elm_except=None):
    choice = random.choice(elm_set)
    while choice == elm_except:
        choice = random.choice(elm_set)
    return choice


def generate_negative_example(
    dataset: pd.DataFrame, entities: pd.DataFrame, times: int = 1
):
    entities_set = entities["entity"].tolist()
    tqdm.pandas(desc="process sample")
    dataset_to_corrupt_head = dataset.sample(frac=0.5, axis=0)
    dataset_to_corrupt_tail = dataset[
        ~dataset.index.isin(dataset_to_corrupt_head.index)
    ]

    # corrupt head
    for index, row in tqdm(
        dataset_to_corrupt_head.iterrows(),
        desc="corrupt head",
        total=len(dataset_to_corrupt_head),
    ):
        for _ in range(times):
            choice = random_choice(entities_set, elm_except=row["head"])
            dataset.loc[len(dataset)] = {
                "head": choice,
                "relation": row["relation"],
                "tail": row["tail"],
                "label": "0",
            }
    # corrupt tail
    for index, row in tqdm(
        dataset_to_corrupt_tail.iterrows(),
        desc="corrupt tail",
        total=len(dataset_to_corrupt_tail),
    ):
        for _ in range(times):
            choice = random_choice(entities_set, elm_except=row["tail"])
            dataset.loc[len(dataset)] = {
                "head": row["head"],
                "relation": row["relation"],
                "tail": choice,
                "label": "0",
            }
    dataset.drop_duplicates()


def process_dataset(dataset_raw, entity2text, relation2text):
    with tqdm(desc="process dataset...", total=4) as process_bar:
        dataset_processed = pd.merge(
            entity2text, dataset_raw, left_on="entity", right_on="head"
        ).loc[:, ["text", "relation", "tail", "label"]]

        process_bar.update(1)

        dataset_processed = pd.merge(
            dataset_processed,
            entity2text,
            left_on="tail",
            right_on="entity",
            suffixes=["_head", "_tail"],
        ).loc[:, ["text_head", "relation", "text_tail", "label"]]
        
        process_bar.update(1)

        dataset_processed = pd.merge(
            dataset_processed, relation2text, left_on="relation", right_on="relation"
        ).loc[:, ["text_head", "text", "text_tail", "label"]]
        
        process_bar.update(1)
        
        dataset_processed.rename(columns={"text": "text_relation"}, inplace=True)
        dataset_processed["text"] = (
            "CLS "
            + dataset_processed["text_head"]
            + " SEP "
            + dataset_processed["text_relation"]
            + " SEP "
            + dataset_processed["text_tail"]
            + " SEP"
        )
        
        process_bar.update(1)

        return dataset_processed.loc[:, ["text", "label"]]


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    entity2text = get_entity2text(data_dir / args.entity2text_file)
    relation2text = get_relation2text(data_dir / args.relation2text_file)
    entities = get_entities(data_dir / args.entities_file)

    process_targets = args.process_targets
    negative_generate_targets = args.negative_generate_targets

    if "train" in process_targets:
        print("process train")
        train = get_dataset(data_dir / args.train_file)

        if "train" in negative_generate_targets:
            print("generate negative example in train")
            generate_negative_example(train, entities, args.pos_neg_ratio)

        processed_train = process_dataset(train, entity2text, relation2text)
        processed_train.to_csv(
            output_dir / f"train{args.output_suffix}.csv", sep=args.output_sep
        )
    if "dev" in process_targets:
        print("process dev")
        dev = get_dataset(data_dir / args.dev_file)

        if "dev" in negative_generate_targets:
            print("generate negative example in dev")
            generate_negative_example(dev, entities, args.pos_neg_ratio)

        processed_dev = process_dataset(dev, entity2text, relation2text)
        processed_dev.to_csv(
            output_dir / f"dev{args.output_suffix}.csv", sep=args.output_sep
        )
    if "test" in process_targets:
        print("process test")
        test = get_dataset(data_dir / args.test_file)

        if "test" in negative_generate_targets:
            print("generate negative example in test")
            generate_negative_example(test, entities, args.pos_neg_ratio)

        processed_dev = process_dataset(test, entity2text, relation2text)
        processed_dev.to_csv(
            output_dir / f"test{args.output_suffix}.csv", sep=args.output_sep
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--train_file", default="train.tsv")
    parser.add_argument("--dev_file", default="dev.tsv")
    parser.add_argument("--test_file", default="test.tsv")
    parser.add_argument("--entity2text_file", default="entity2text.txt")
    parser.add_argument("--relation2text_file", default="relation2text.txt")
    parser.add_argument("--entities_file", default="entities.txt")
    parser.add_argument("--pos_neg_ratio", type=int, default=5)
    parser.add_argument("--process_targets", action='store', type=str, nargs="*", default=["train", "dev", "test"])
    parser.add_argument("--negative_generate_targets", action='store', type=str, nargs="*", default=["train", "dev"])
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--output_suffix", default="_processed")
    parser.add_argument("--output_sep", default="\t")

    args = parser.parse_args()

    main(args)
