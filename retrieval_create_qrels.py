import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path

import pandas as pd
import simple_parsing

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    papers_file: Path = field(default=Path("data/papers.jsonl"))
    qa_file: Path = field(default=Path("data/qa.jsonl"))
    output_dir: Path = field(default=Path("out"))


def main(args):

    # data loading
    qa_df = pd.read_json(args.qa_file, lines=True)
    papers_df = pd.read_json(args.papers_file, lines=True)

    paragraph_qrels = defaultdict(dict)
    sentence_qrels = defaultdict(dict)
    for _, qa in qa_df.iterrows():

        if qa.answer_evidence_mapped is None:
            # No Answer Evidence has been annotated for this Question
            continue

        qidx = qa.question_id
        for ae in qa.answer_evidence_mapped:
            lidx = ae["idx"]
            for idx in lidx:

                if idx is None:
                    # Answer evidence that has no match in the extracted text
                    continue

                try:
                    pidx, sidx = papers_df[
                        (papers_df.paper_id == qa.paper_id) & (papers_df.idx == idx)
                    ][["pidx", "sidx"]].values[0]
                except Exception as e:
                    logger.error(f"Error: {e}")
                    logger.error(f"Question ID: {qidx}, Paper ID: {qa.paper_id}, idx: {idx}")
                    raise e
                paragraph_qrels[qidx][str(pidx)] = 1
                sentence_qrels[qidx][f"{pidx}/{sidx}"] = 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, qrels in zip(
        ["sentences", "paragraphs"], [sentence_qrels, paragraph_qrels]
    ):
        with open(args.output_dir / f"qrels.{name}.json", "w", encoding="utf-8") as f:
            json.dump(qrels, f, indent=2)


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    logger.info(args)
    main(args)
