import random
import pandas as pd
from tqdm import tqdm
from flair.data import Sentence


def process_dataset(dataset_raw, entity2text, relation2text, hidde_bar: bool = False):
    with tqdm(desc="process dataset...", total=4, disable=hidde_bar) as process_bar:
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
            "[CLS] "
            + dataset_processed["text_head"]
            + " [SEP] "
            + dataset_processed["text_relation"]
            + " [SEP] "
            + dataset_processed["text_tail"]
            + " [SEP]"
        )

        process_bar.update(1)

        return dataset_processed.loc[:, ["text", "label"]]


def get_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path, sep="\t", header=None)
    dataset.columns = ["head", "relation", "tail"]
    dataset["label"] = "1"
    return dataset


def get_entity2text(path):
    data = pd.read_csv(path, sep="\t", header=None)
    data.columns = ["entity", "text"]
    return data


def get_relation2text(path):
    data = pd.read_csv(path, sep="\t", header=None)
    data.columns = ["relation", "text"]
    return data


def get_entities(path):
    data = pd.read_csv(path, sep="\t", header=None)
    data.columns = ["entity"]
    return data


def random_choice(elm_set: list, elm_except=None):
    choice = random.choice(elm_set)
    while choice == elm_except:
        choice = random.choice(elm_set)
    return choice


def generate_negative_example(
    dataset: pd.DataFrame,
    entities: pd.DataFrame,
    times: int = 1,
    hidde_bar: bool = False,
):
    entities_set = entities["entity"].tolist()
    tqdm.pandas(desc="process sample", disable=hidde_bar)
    dataset_to_corrupt_head = dataset.sample(frac=0.5, axis=0)
    dataset_to_corrupt_tail = dataset[
        ~dataset.index.isin(dataset_to_corrupt_head.index)
    ]

    # corrupt head
    for index, row in tqdm(
        dataset_to_corrupt_head.iterrows(),
        desc="corrupt head",
        total=len(dataset_to_corrupt_head),
        disable=hidde_bar,
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
        disable=hidde_bar,
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


def generate_sentence_eval_examples(
    row: pd.Series,
    entity2text: pd.DataFrame,
    relation2text: pd.DataFrame,
    entities: pd.DataFrame,
    corrupt: str = "left",
):
    if corrupt == "left":
        example_table = pd.DataFrame(columns=["head", "relation", "tail", "label"])
        # example_table.loc[0] = row
        example_table["head"] = entities["entity"]
        true_head_idx = example_table[
            example_table["head"] == row["head"]
        ].index.to_list()[0]
        example_table.loc[0], example_table.loc[true_head_idx] = (
            example_table.loc[true_head_idx].copy(),
            example_table.loc[0].copy(),
        )
        example_table["relation"] = row["relation"]
        example_table["tail"] = row["tail"]
        example_table["label"] = "0"
        example_table.drop_duplicates()
    elif corrupt == "right":
        example_table = pd.DataFrame(columns=["head", "relation", "tail"])
        # example_table.loc[0] = row
        example_table["tail"] = entities["entity"]
        true_tail_idx = example_table[
            example_table["tail"] == row["tail"]
        ].index.to_list()[0]
        example_table.loc[0], example_table.loc[true_tail_idx] = (
            example_table.loc[true_tail_idx].copy(),
            example_table.loc[0].copy(),
        )
        example_table["relation"] = row["relation"]
        example_table["head"] = row["head"]
        example_table["label"] = "0"
        example_table.drop_duplicates()
    else:
        raise Exception()

    processed_table = process_dataset(
        example_table, entity2text, relation2text, hidde_bar=True
    )

    for index, row in processed_table.iterrows():
        yield Sentence(row["text"])


def generate_eval_examples(
    dataset_raw: pd.DataFrame,
    entity2text: pd.DataFrame,
    relation2text: pd.DataFrame,
    entities: pd.DataFrame,
    corrupt: str = "left",
):
    for index, row in dataset_raw.iterrows():
        yield generate_sentence_eval_examples(
            row, entity2text, relation2text, entities, corrupt
        )
