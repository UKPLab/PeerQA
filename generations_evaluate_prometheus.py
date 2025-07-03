# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "prometheus-eval",
#     "simple-parsing",
#     "pandas",
#     "tqdm",
#     "setuptools",
#     "vllm==0.9.1",
# ]
# ///
import logging
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import Optional

import pandas as pd
import simple_parsing
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from prometheus_eval.vllm import VLLM
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    generation_file: Path
    qa_file: Path = field(default=Path("data/qa.jsonl"))
    papers_file: Path = field(default=Path("data/papers.jsonl"))
    qa_augmented_answers_file: Optional[Path] = field(
        default=Path("data/qa-augmented-answers.jsonl")
    )
    output_dir: Path = field(default=Path("out"))
    model: str = "prometheus-eval/prometheus-7b-v2.0"


def _answer_evidence_concat(answer_evidence: list[dict]) -> str:
    """
    Concatenate the evidence sentences,
    if they are mapped to a extracted sentence from the paper.
    """
    if answer_evidence is None:
        return None
    answer_evidence = sorted(answer_evidence, key=lambda x: x["idx"])
    concat = " \n ".join(ae["sentence"] for ae in answer_evidence)
    return concat


def _answer_evidence_concat_paragraphs(paper_df, paper_id, answer_evidence_mapped):
    """
    Concatenate the evidence paragraphs,
    if they are mapped to a extracted sentence from the paper.
    """
    idx = [i for ae in answer_evidence_mapped for i in ae["idx"]]
    paper_sents = paper_df[(paper_df.paper_id == paper_id) & (paper_df.idx.isin(idx))]
    pidx = paper_sents.pidx.unique()
    paper_paras = paper_df[
        (paper_df.paper_id == paper_id) & (paper_df.pidx.isin(pidx))
    ].sort_values("pidx")
    concat = " \n ".join(paper_paras.content)
    return concat


def main(args):

    out_file = (
        args.output_dir / f"metrics-generations-prometheus-{args.generation_file.name}"
    )

    gen_df = pd.read_json(args.generation_file, lines=True)
    qa_df = pd.read_json(args.qa_file, lines=True)
    paper_df = pd.read_json(args.papers_file, lines=True)

    # concatenate the 'content' column of paper_df per paper_id
    full_text_df = paper_df.groupby("paper_id")["content"].apply(" ".join).reset_index()
    full_text_df.rename(columns={"content": "full_text"}, inplace=True)

    # Preprocess generations
    # Add answer evidence and answer free form to the generated data
    gen_df = pd.merge(
        gen_df,
        qa_df[
            [
                "paper_id",
                "question_id",
                "answerable_mapped",
                "answer_evidence_mapped",
                "answer_free_form",
            ]
        ],
        on=["paper_id", "question_id"],
        how="left",
    )

    if args.qa_augmented_answers_file is not None:
        # if provided, add `augmented_answer_free_form` column to qa_df
        qa_augmented_answers_df = pd.read_json(
            args.qa_augmented_answers_file, lines=True
        )
        gen_df = pd.merge(
            gen_df,
            qa_augmented_answers_df[
                ["paper_id", "question_id", "augmented_answer_free_form"]
            ],
            on=["paper_id", "question_id"],
            how="left",
        )

    # drop rows where answerable_mapped is False or None
    gen_df = gen_df[gen_df.answerable_mapped == True]

    # Add the full text of the paper to the generated data
    gen_df = pd.merge(
        gen_df,
        full_text_df[["paper_id", "full_text"]],
        on="paper_id",
        how="left",
    )

    # Concatenate the evidence sentences
    gen_df["answer_evidence_mapped_concat"] = gen_df.answer_evidence_mapped.apply(
        _answer_evidence_concat
    )

    # Concatenate the evidence paragraphs
    gen_df["answer_evidence_para_concat"] = gen_df.apply(
        lambda x: _answer_evidence_concat_paragraphs(
            paper_df, x.paper_id, x.answer_evidence_mapped
        ),
        axis=1,
    )

    context_generation_cols = [
        ("answer_free_form", "generation"),
        ("augmented_answer_free_form", "generation"),
    ]
    col_abbr = {
        "answer_evidence_mapped_concat": "aem",
        "answer_evidence_para_concat": "aep",
        "answer_free_form": "ff",
        "augmented_answer_free_form": "aff",
        "generation": "gen",
        "full_text": "ft",
    }

    cirteria_prometheus_params = {
        "relevance": {
            "instruction": "Your task is to evaluate the generated answer against the reference answer for the question: {{query}}",
            "rubric_data": {
                "criteria": "Relevancy",
                "score1_description": "The answer is off-topic or irrelevant to the question asked",
                "score2_description": "The answer is includes some relevant information but often contains unrelated details.",
                "score3_description": "The answer is generally relevant to the question but occasionally includes extraneous or off-topic details.",
                "score4_description": "The answer is mostly relevant to the question, with minimal unrelated information.",
                "score5_description": "The answer is highly relevant to the question, addressing it directly and thoroughly without including unnecessary information.",
            },
        },
        "correctness": {
            "instruction": "Your task is to evaluate the generated answer against the reference answer for the question: {{query}}",
            "rubric_data": {
                "criteria": "Correctness",
                "score1_description": "The answer is not relevant to the question and does not align with the reference answer.",
                "score2_description": "The answer is relevant to the question but deviates significantly from the reference answer.",
                "score3_description": "The answer is relevant to the question and generally aligns with the reference answer but has errors or omissions.",
                "score4_description": "The answer is relevant to the question and closely matches the reference answer but is less concise or clear.",
                "score5_description": "The answer is highly relevant, fully accurate, and matches the reference answer in both content and clarity.",
            },
        },
    }

    model = VLLM(model=args.model)
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    for context_col, generation_col in context_generation_cols:

        eval_prefix = f"{col_abbr[context_col]}-{col_abbr[generation_col]}"

        mask = gen_df[context_col].notnull() & gen_df[generation_col].notnull()
        paper_ids, question_ids, questions, reference_answers, responses = gen_df[mask][
            ["paper_id", "question_id", "question", context_col, generation_col]
        ].values.T

        reference_answers = list(reference_answers)
        responses = list(responses)

        for criteria, prometheus_params in cirteria_prometheus_params.items():
            instructions = []
            for paper_id, question_id, question, reference_answer, response in zip(
                paper_ids, question_ids, questions, reference_answers, responses
            ):

                instructions.append(
                    prometheus_params["instruction"].format(query=question)
                )
                # reference_answers.append(reference_answer)
                # responses.append(response)

            assert (
                len(instructions) == len(responses) == len(reference_answers)
            ), f"{len(instructions)} != {len(responses)} != {len(reference_answers)}"

            assert all(len(r) > 0 for r in responses), "There is an empty response."
            assert all(
                len(r) > 0 for r in reference_answers
            ), "There is an empty reference answer."

            feedbacks, scores = judge.absolute_grade(
                instructions=instructions,
                responses=responses,
                reference_answers=reference_answers,
                rubric=SCORE_RUBRIC_TEMPLATE.format(**prometheus_params["rubric_data"]),
            )

            gen_df.loc[mask, f"{eval_prefix}-{criteria}-prometheus-scores"] = scores
            gen_df.loc[mask, f"{eval_prefix}-{criteria}-prometheus-feedback"] = (
                feedbacks
            )

    metric_cols = [c for c in gen_df.columns if "prometheus" in c]
    gen_df[["paper_id", "question_id", *metric_cols]].to_json(
        out_file, orient="records", lines=True
    )


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
