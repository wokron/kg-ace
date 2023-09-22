import argparse
from pathlib import Path
from model import KGClassifier
from metrics import (
    get_rank,
    calc_hit_at_k,
    calc_mean_rank,
    calc_mean_reciprocal_ranking,
)
from process import (
    get_entity2text,
    get_relation2text,
    get_entities,
    get_dataset,
    generate_eval_examples,
)
import numpy as np
from tqdm import tqdm


def main(args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    entity2text = get_entity2text(data_dir / args.entity2text_file)
    relation2text = get_relation2text(data_dir / args.relation2text_file)
    entities = get_entities(data_dir / args.entities_file)
    dataset = get_dataset(data_dir / args.src_dataset)

    eval_examples = generate_eval_examples(
        dataset,
        entity2text,
        relation2text,
        entities,
        corrupt="right",
    )

    model = KGClassifier.load(args.model_path)

    ranks = []
    for eval_example in tqdm(
        eval_examples, desc="iterate over the dataset", total=len(dataset)
    ):
        pos_s = next(eval_example)
        neg_ss = [
            s
            for s in tqdm(
                eval_example,
                desc="generate neg examples",
                total=len(entities) - 1,
                leave=False,
            )
        ]

        model.predict([pos_s] + neg_ss, return_probabilities_for_all_classes=True)

        rank = get_rank(pos_s, neg_ss)
        ranks.append(rank)
    ranks = np.array(ranks)
    metrics = {}
    for k in [1, 3, 10]:
        hit_at_k = calc_hit_at_k(ranks, k)
        metrics[f"hit@{k}"] = hit_at_k
    metrics["mr"] = calc_mean_rank(ranks)
    metrics["mrr"] = calc_mean_reciprocal_ranking(ranks)

    with open(output_dir / "eval_result.txt", "w") as f:
        for k in metrics.keys():
            output = f"{k} = {metrics[k]}"
            print(output)
            f.write(output + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--src_dataset", default="test.tsv")
    parser.add_argument("--entity2text_file", default="entity2text.txt")
    parser.add_argument("--relation2text_file", default="relation2text.txt")
    parser.add_argument("--entities_file", default="entities.txt")
    parser.add_argument("--model_path", default="output/kg-ace/best-model.pt")
    parser.add_argument("--output_dir", default="output")

    args = parser.parse_args()

    main(args)
