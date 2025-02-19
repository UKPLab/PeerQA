# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "alignscore",
#     "datasets<3.0.0",
#     "en-core-web-sm",
#     "pandas",
#     "rouge-score==0.1.2",
#     "simple-parsing",
#     "tqdm",
#     "transformers<=4.47",
# ]
#
# [tool.uv.sources]
# alignscore = { git = "https://github.com/yuh-zha/AlignScore" }
# en-core-web-sm = { url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz" }
# ///
import logging
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import simple_parsing
from datasets import load_metric
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    generation_file: Path
    alignscore_checkpoint_file: str = (
        "AlignScore-large.ckpt"  # Filepath to the AlignScore checkpoint, download from https://huggingface.co/yzha/AlignScore/tree/main
    )
    qa_file: Path = field(default=Path("data/qa.jsonl"))
    papers_file: Path = field(default=Path("data/papers.jsonl"))
    qa_augmented_answers_file: Optional[Path] = field(
        default=Path("data/qa-augmented-answers.jsonl")
    )
    output_dir: Path = field(default=Path("out"))
    override: bool = False
    extend: bool = False
    alignscore_evaluation_mode: str = "nli_sp"
    alignscore_model_size: Literal["base", "large"] = "large"
    alignscore_batch_size: int = 8
    skip_alignscore: bool = False


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


def main(args: Args):

    out_file = args.output_dir / f"metrics-{args.generation_file.name}"
    if out_file.exists() and not args.override and not args.extend:
        logger.info(
            f"{out_file} exists. Skipping, because `override=False` and `extend=False`."
        )
    elif out_file.exists() and args.extend:
        df_computed = pd.read_json(out_file, lines=True)
        already_computed_metrics = list(df_computed.columns)
        already_computed_metrics.remove("paper_id")
        already_computed_metrics.remove("question_id")
        logger.info(f"Found already computed metrics: {already_computed_metrics}")
    else:
        already_computed_metrics = []

    # Load the data
    logger.info(f"Loading data from {args.generation_file}.")
    gen_df = pd.read_json(args.generation_file, lines=True)
    qa_df = pd.read_json(args.qa_file, lines=True)
    paper_df = pd.read_json(args.papers_file, lines=True)

    # concatenate the 'content' column of paper_df per paper_id
    full_text_df = paper_df.groupby("paper_id")["content"].apply(" ".join).reset_index()
    full_text_df.rename(columns={"content": "full_text"}, inplace=True)

    # Preprocess generations
    logger.info("Preprocessing generations.")
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
        logger.info(
            f"Loaded augmented answers from {args.qa_augmented_answers_file}. {qa_augmented_answers_df.columns=}"
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

    # maybe merge with the previously computed
    if args.extend and already_computed_metrics:
        gen_df = pd.merge(
            gen_df, df_computed, on=["paper_id", "question_id"], how="left"
        )

    logger.info("Computing metrics.")
    # Compute the metrics for the different context-generation pairs
    context_generation_cols = [
        # ("answer_evidence_mapped_concat", "answer_free_form"),
        # ("answer_evidence_mapped_concat", "generation"),
        ("answer_evidence_para_concat", "answer_free_form"),
        ("answer_evidence_para_concat", "generation"),
        ("answer_free_form", "generation"),
        # ("full_text", "answer_free_form"),
        # ("full_text", "generation"),
    ]
    if args.qa_augmented_answers_file is not None:
        context_generation_cols.extend(
            [
                # ("answer_evidence_mapped_concat", "augmented_answer_free_form"),
                ("answer_evidence_para_concat", "augmented_answer_free_form"),
                ("answer_free_form", "augmented_answer_free_form"),
                ("augmented_answer_free_form", "answer_free_form"),
                ("augmented_answer_free_form", "generation"),
            ]
        )
    logger.info(f"Evaluating over columns: {context_generation_cols=}")

    col_abbr = {
        "answer_evidence_mapped_concat": "aem",
        "answer_evidence_para_concat": "aep",
        "answer_free_form": "ff",
        "augmented_answer_free_form": "aff",
        "generation": "gen",
        "full_text": "ft",
    }
    if not args.skip_alignscore:
        logger.info(f"Loading AlignScore.")
        import torch  # isort: skip
        from alignscore import AlignScore  # isort: skip

        align_evaluator = AlignScore(
            model=f"roberta-{args.alignscore_model_size}",
            batch_size=args.alignscore_batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            ckpt_path=args.alignscore_checkpoint_file,
            evaluation_mode=args.alignscore_evaluation_mode,
            verbose=True,
        )

    rouge_evaluator = load_metric(
        "rouge",
        experiment_id=f"rouge-{args.generation_file.name}",
        trust_remote_code=True,
    )

    # times 2 for rouge and alignscore
    with tqdm(total=len(context_generation_cols) * 2, ncols=80) as pbar:
        for context_col, generation_col in context_generation_cols:

            eval_prefix = f"{col_abbr[context_col]}-{col_abbr[generation_col]}"

            mask = gen_df[context_col].notnull() & gen_df[generation_col].notnull()
            paper_ids, question_ids, contexts, generations = gen_df[mask][
                ["paper_id", "question_id", context_col, generation_col]
            ].values.T

            pbar.set_description(
                f"{col_abbr[context_col]} -> {col_abbr[generation_col]} R"
            )
            if args.override or (
                f"{eval_prefix}-rougel-fmeasure" not in already_computed_metrics
            ):

                for use_stemmer in [True, False]:
                    rouge_scores = rouge_evaluator.compute(
                        predictions=contexts,
                        references=generations,
                        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
                        use_aggregator=False,  # get individual scores per example
                        use_stemmer=use_stemmer,
                    )
                    stemmer_key = "stemmer" if use_stemmer else "no_stemmer"
                    for rouge_key, rouge_value in rouge_scores.items():
                        rouge_key = rouge_key.lower()
                        gen_df.loc[
                            mask, f"{eval_prefix}-{rouge_key}-{stemmer_key}-precision"
                        ] = [r.precision for r in rouge_value]
                        gen_df.loc[
                            mask, f"{eval_prefix}-{rouge_key}-{stemmer_key}-recall"
                        ] = [r.recall for r in rouge_value]
                        gen_df.loc[
                            mask, f"{eval_prefix}-{rouge_key}-{stemmer_key}-fmeasure"
                        ] = [r.fmeasure for r in rouge_value]
            pbar.update(1)

            pbar.set_description(
                f"{col_abbr[context_col]} -> {col_abbr[generation_col]} AS"
            )
            if not args.skip_alignscore and (
                args.override
                or (f"{eval_prefix}-alignscore" not in already_computed_metrics)
            ):
                align_scores = align_evaluator.score(contexts, generations)
                gen_df.loc[mask, f"{eval_prefix}-alignscore"] = align_scores
            pbar.update(1)

    metric_cols = [c for c in gen_df.columns if "rouge" in c or "alignscore" in c]
    gen_df[["paper_id", "question_id", *metric_cols]].to_json(
        out_file, orient="records", lines=True
    )


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
