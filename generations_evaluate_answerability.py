import logging
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path

import pandas as pd
import simple_parsing
from sklearn.metrics import classification_report
from tqdm.contrib.logging import logging_redirect_tqdm

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    generation_file: Path
    output_dir: Path = field(default=Path("out"))
    qa_file: Path = field(default=Path("data/qa.jsonl"))
    override: bool = False
    no_answer_string: str = "No Answer"


def main(args: Args):

    out_file = (
        args.output_dir / f"metrics-answerability-{args.generation_file.name}.json"
    )
    if out_file.exists() and not args.override:
        logger.info(f"Metrics file {out_file} already exists. Skipping.")
        return

    # Load the data
    logger.info(f"Loading data from {args.generation_file} and {args.qa_file}.")
    gen_df = pd.read_json(args.generation_file, lines=True)
    # drop answerable column, we will get the label from qa_df
    gen_df = gen_df.drop(columns=["answerable"])
    qa_df = pd.read_json(args.qa_file, lines=True)
    logger.info(f"qa_df shape: {qa_df.shape}")

    logger.info("Preprocessing generations.")
    # Add answer evidence and answer free form to the generated data
    gen_df = pd.merge(
        gen_df,
        qa_df[["paper_id", "question_id", "answerable_mapped"]],
        on=["paper_id", "question_id"],
        how="left",
    )

    # Drop rows with missing answerable_mapped, as we can't evaluate them
    gen_df = gen_df.dropna(subset=["answerable_mapped"])
    logger.info(f"gen_df shape: {gen_df.shape}")

    y_pred = 1 - gen_df.generation.str.contains(args.no_answer_string).astype(int)
    y_true = gen_df.answerable_mapped.astype(int)
    logger.info(f"y_pred shape: {y_pred.shape} {y_pred.sum()=}")
    logger.info(f"y_true shape: {y_true.shape} {y_true.sum()=}")

    clf_report_kwargs = dict(
        y_true=y_true,
        y_pred=y_pred,
        target_names=["Unanswerable", "Answerable"],
        zero_division=0,
    )
    clf_report_str = classification_report(**clf_report_kwargs)
    logger.info("\n" + clf_report_str)

    logger.info("Saving classification report.")
    clf_report = classification_report(**clf_report_kwargs, output_dict=True)
    pd.DataFrame(clf_report).to_json(out_file, indent=2)


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
