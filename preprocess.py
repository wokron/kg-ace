from pathlib import Path
import argparse
from process import (
    process_dataset,
    get_entity2text,
    get_relation2text,
    get_entities,
    get_dataset,
    generate_negative_example,
)


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
    parser.add_argument("--pos_neg_ratio", type=int, default=1)
    parser.add_argument(
        "--process_targets",
        action="store",
        type=str,
        nargs="*",
        default=["train", "dev", "test"],
    )
    parser.add_argument(
        "--negative_generate_targets",
        action="store",
        type=str,
        nargs="*",
        default=["train", "dev"],
    )
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--output_suffix", default="_processed")
    parser.add_argument("--output_sep", default="\t")

    args = parser.parse_args()

    main(args)
