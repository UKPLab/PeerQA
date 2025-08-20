# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pyserini",
#     "simple-parsing",
# ]
# ///
import json
import logging
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import Literal

import simple_parsing
from pyserini.search.lucene import LuceneSearcher
from tqdm.contrib.logging import logging_redirect_tqdm

from peerqa.data_loader import QuestionLoader
from peerqa.utils import url_save_str

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    qa_file: Path = field(default=Path("data/qa.jsonl"))
    output_dir: Path = field(default=Path("out"))
    granularity: Literal["sentences", "paragraphs"] = "sentences"


def main(args: Args):

    qa_loader = QuestionLoader(args.qa_file)

    run = {}

    for paper_id, question_ids, questions in qa_loader.questions_with_answer_evidence():

        if len(questions) == 0:
            continue

        index_dir = (
            args.output_dir
            / "pyserini"
            / "indexes"
            / f"bm25-{args.granularity}"
            / url_save_str(paper_id)
        )
        index_dir.mkdir(parents=True, exist_ok=True)
        searcher = LuceneSearcher(str(index_dir))

        hits = searcher.batch_search(questions, qids=question_ids, k=1000)
        for question_id, question_hits in hits.items():
            run[question_id] = {}
            for hit in question_hits:
                run[question_id][hit.docid] = hit.score

        with open(
            args.output_dir / f"run-{args.granularity}-bm25-sparse.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(run, f, indent=2)


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
