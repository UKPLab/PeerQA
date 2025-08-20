import json
import logging
from collections import defaultdict
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
    output_type: Literal["bm25", "dense"] = "bm25"
    granularity: Literal["sentences", "paragraphs"] = "sentences"
    template: str = None


def main(args: Args):

    paper_loader = PaperLoader(args.papers_file)
    qa_loader = QuestionLoader(args.qa_file)

    document_subdir = f"documents-{args.output_type}-{args.granularity}"
    if args.template is not None:
        template_hash = url_save_hash(args.template)
        document_subdir += f"-{template_hash}"

    pyserini_data = defaultdict(list)
    paper_to_topics = defaultdict(list)
    for paper_id, document_ids, documents in paper_loader(
        granularity=args.granularity, template=args.template
    ):
        question_ids, questions = qa_loader.questions_by_paper_id_with_answer_evidence(
            paper_id=paper_id
        )

        if len(questions) == 0:
            # No Questions (with Answer Evidence) for this Paper
            continue

        # build topics file
        for question_id, question in zip(question_ids, questions):
            paper_to_topics[paper_id].append([question_id, question])

        # create directory to store documents in
        paper_index_path = (
            args.output_dir / "pyserini" / document_subdir / url_save_str(paper_id)
        )
        paper_index_path.mkdir(parents=True, exist_ok=True)

        # dump documents per paper for bm25 or dense rertrieval
        if args.output_type == "bm25":
            # for content in paper["paper"]["content"]:
            for document_id, document in zip(document_ids, documents):
                pyserini_data[paper_id].append(
                    {
                        "id": document_id,
                        "contents": document,
                    }
                )

            with open(paper_index_path / "documents.json", "w", encoding="utf-8") as f:
                json.dump(pyserini_data[paper_id], f, indent=2)

        elif args.output_type == "dense":
            for document_id, document in zip(document_ids, documents):
                pyserini_data[paper_id].append(
                    {
                        "id": document_id,
                        "text": document,
                    }
                )
            # dump to jsonl
            with open(paper_index_path / "documents.json", "w", encoding="utf-8") as f:
                for doc in pyserini_data[paper_id]:
                    f.write(json.dumps(doc) + "\n")
        else:
            raise ValueError(f"Unknown output type: {args.output_type}")

    # dump collected topics
    topics_dir = args.output_dir / "pyserini" / f"topics"
    topics_dir.mkdir(parents=True, exist_ok=True)
    for paper_id, topics in paper_to_topics.items():
        with open(
            topics_dir / f"{url_save_str(paper_id)}.tsv",
            "w",
            encoding="utf-8",
        ) as f:
            for topic in topics:
                f.write("\t".join(topic) + "\n")


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
