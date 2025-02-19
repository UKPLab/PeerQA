import logging
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import Literal

import simple_parsing
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from peerqa.data_loader import PaperLoader, QuestionLoader
from peerqa.utils import url_save_hash, url_save_str

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    output_dir: Path = field(default=Path("out"))
    qa_file: Path = field(default=Path("data/qa.jsonl"))
    papers_file: Path = field(default=Path("data/papers.jsonl"))
    granularity: Literal["sentences", "paragraphs"] = "sentences"
    template: str = None


def main(args):

    paper_loader = PaperLoader(args.papers_file)
    qa_loader = QuestionLoader(args.qa_file)

    subdir = f"colbert-{args.granularity}"
    if args.template is not None:
        template_hash = url_save_hash(args.template)
        logger.info(f"Adding template hash {template_hash} to subdir.")
        subdir += f"-{template_hash}"

    for paper_id, document_ids, documents in paper_loader(
        granularity=args.granularity, template=args.template
    ):
        question_ids, questions = qa_loader.questions_by_paper_id_with_answer_evidence(
            paper_id=paper_id
        )

        if len(questions) == 0:
            continue

        # Queries: each line is qid \t query text.
        queries_path = (
            args.output_dir / subdir / f"{url_save_str(paper_id)}" / "queries.tsv"
        )
        queries_path.parent.mkdir(parents=True, exist_ok=True)
        with open(queries_path, "w") as fh:
            for question_id, question in zip(question_ids, questions):
                fh.write(f"{question_id}\t{question}\n")

        # Collection: each line is pid \t passage text.
        collection_path = (
            args.output_dir / subdir / f"{url_save_str(paper_id)}" / "collection.tsv"
        )
        collection_path.parent.mkdir(parents=True, exist_ok=True)
        with open(collection_path, "w") as fh:
            for document_id, document in zip(document_ids, documents):
                fh.write(f"{document_id}\t{document}\n")


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
