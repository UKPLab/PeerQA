import json
import logging
from dataclasses import dataclass, field
from functools import partial
from logging.config import fileConfig
from pathlib import Path
from typing import Literal, Union

import numpy as np
import simple_parsing
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from tqdm.contrib.logging import logging_redirect_tqdm

from peerqa.data_loader import HydeLoader, PaperLoader, QuestionLoader
from peerqa.dense_wrappers import SequenceClassification, Splade, HFBase
from peerqa.utils import url_save_hash, url_save_str

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    query_model: str
    document_model: str = None
    model_cls: Literal["st", "cross", "hf", "splade", "sc"] = "st"
    output_dir: Path = field(default=Path("out"))
    qa_file: Path = field(default=Path("data/qa.jsonl"))
    papers_file: Path = field(default=Path("data/papers.jsonl"))
    sim_fn: Literal["cos", "dot", "cross"] = "dot"
    batch_size: int = 32
    pooling: str = None
    granularity: Literal["sentences", "paragraphs"] = "sentences"
    template: str = None
    override: bool = False
    hyde_file: Path = None
    hyde_add_question_embeddings_to_average: bool = True
    include_unanswerable: bool = False


def init_model(model, pooling):

    if args.model_cls == "st":
        model_cls = SentenceTransformer
    elif args.model_cls == "cross":
        model_cls = CrossEncoder
    elif args.model_cls == "hf":
        model_cls = partial(HFBase, pooling=pooling)
    elif args.model_cls == "splade":
        model_cls = partial(Splade, pooling=pooling)
    elif args.model_cls == "sc":
        model_cls = SequenceClassification

    return model_cls(model)


def process_hyde(
    args, question_embeddings, question_ids, hyde_loader, document_model, n=8
):
    """Process the hyde embeddings for the given questions.

    1. Compute Hyde passage embeddings
    2. Concate question and hyde embeddings
    3. Average over the passage dimension

    """
    # Get the Hyde passages
    hyde_passages = []
    for question_id in question_ids:
        hyde_passages.extend(hyde_loader.passages_by_question_id(question_id))
    hyde_embeddings = document_model.encode(
        hyde_passages, show_progress_bar=False, batch_size=args.batch_size
    )

    # Reshape to (n_questions, n_passages, embedding_dim)
    hyde_embeddings = hyde_embeddings.reshape(len(question_ids), n, -1)

    if args.hyde_add_question_embeddings_to_average:
        question_embeddings = question_embeddings.reshape(len(question_ids), 1, -1)

        # Append question_embeddings to the end of the passage embeddings
        hyde_question_embeddings = np.concatenate(
            [hyde_embeddings, question_embeddings], axis=1
        )

    # Average over the passage dimension
    hyde_question_embeddings = np.mean(hyde_question_embeddings, axis=1)

    return hyde_question_embeddings


def main(args: Args):

    model_str = url_save_str(args.query_model)
    if args.use_hyde:
        model_str += "-hyde"
    if args.include_unanswerable:
        model_str += "-unsanswerable"

    if args.template is not None:
        template_hash = url_save_hash(args.template)
        out_path_suffix = f"-{template_hash}"
    else:
        out_path_suffix = ""

    out_path = (
        args.output_dir
        / f"run-{args.granularity}-{model_str}-{args.sim_fn}{out_path_suffix}.json"
    )
    if (not args.override) and out_path.exists():
        logger.info(f"Skipping since {out_path=} already exists.")
        return

    logger.info(f"Will write results to {out_path=}.")

    # Init Model
    query_model = init_model(model=args.query_model, pooling=args.pooling)
    if args.document_model is None:
        document_model = query_model
    else:
        document_model = init_model(model=args.document_model, pooling=args.pooling)

    # Set the Similarity function (dense retrieval only)
    if args.sim_fn == "cos":
        sim_fn = util.pytorch_cos_sim
    elif args.sim_fn == "dot":
        sim_fn = util.dot_score
    elif args.sim_fn == "cross":
        pass
    else:
        raise ValueError(args.sim_fn)

    # Init Data
    qa_loader = QuestionLoader(args.qa_file)
    paper_loader = PaperLoader(args.papers_file)
    if args.hyde_file is not None:
        hyde_loader = HydeLoader(str(args.hyde_file))

    with open(args.output_dir / f"qrels.{args.granularity}.json", "r") as f:
        qrels = json.load(f)

    run = {}
    for paper_id, document_ids, documents in paper_loader(
        granularity=args.granularity, template=args.template
    ):
        question_ids, questions = qa_loader.questions_by_paper_id_with_answer_evidence(
            paper_id=paper_id, include_unanswerable=args.include_unanswerable
        )

        if len(questions) == 0:
            # No Questions (with Answer Evidence) for this Paper
            continue

        # Assert that all question ids have some entry in qrels
        assert (
            all(qid in qrels.keys() for qid in question_ids)
            or args.include_unanswerable
        ), list(filter(lambda qid: qid not in qrels.keys(), question_ids))

        if isinstance(query_model, (SentenceTransformer, HFBase)):
            question_embeddings = query_model.encode(
                questions, show_progress_bar=False, batch_size=args.batch_size
            )
            if args.use_hyde:
                question_embeddings = process_hyde(
                    args=args,
                    question_embeddings=question_embeddings,
                    question_ids=question_ids,
                    hyde_loader=hyde_loader,
                    document_model=document_model,
                )

            document_embeddings = document_model.encode(
                documents, show_progress_bar=False, batch_size=args.batch_size
            )

            scores = sim_fn(question_embeddings, document_embeddings)

        elif isinstance(query_model, (CrossEncoder, SequenceClassification)):
            pairs = [[q, d] for q in questions for d in documents]
            scores = query_model.predict(
                pairs, show_progress_bar=False, batch_size=args.batch_size
            ).reshape(len(questions), len(documents))
        else:
            raise RuntimeError(type(query_model))

        for i, question_id in enumerate(question_ids):
            run[question_id] = {}

            for didx, score in zip(document_ids, scores[i]):
                run[question_id][didx] = score.item()

    with open(out_path, "w") as f:
        json.dump(run, f, indent=2)


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    args.use_hyde = args.hyde_file is not None
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
