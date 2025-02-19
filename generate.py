# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "setuptools",
#     "simple-parsing",
#     "vllm==0.4.2",
# ]
# ///
import json
import logging
import random
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import Literal

import pandas as pd
import simple_parsing
from tqdm.contrib.logging import logging_redirect_tqdm
from vllm import LLM, SamplingParams

from peerqa.data_loader import PaperLoader, QuestionLoader
from peerqa.generate_utils import (make_prompts_full_text, make_prompts_rag,
                                   process_outputs)
from peerqa.prompts import PROMPTS

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:

    retrieval_file: Path = field(
        default=Path("out/run-paragraphs-naver_splade-v3-unsanswerable-dot.json")
    )
    output_dir: Path = field(default=Path("out"))
    qa_file: Path = field(default=Path("data/qa.jsonl"))
    papers_file: Path = field(default=Path("data/papers.jsonl"))
    prompt_selection: Literal[
        "answerability-full-text",
        "answerability-rag",
        "full-text",
        "rag",
        "prompt_template",
    ] = "rag"
    prompt_template: str = None
    sort_by: Literal["score", "document"] = "score"
    model: Literal[
        "command-r-v01",
        "llama-8B-instruct-32k",
        "llama-8B-instruct",
        "mistral-7B-instruct-v02",
    ] = "llama-8B-instruct"
    context_setting: int | str = None

    def __post_init__(self):

        if "full-text" in self.prompt_selection:
            if self.context_setting is None:
                self.context_setting = "full-text"
            if self.context_setting != "full-text":
                raise ValueError(
                    f"Invalid context_setting: {self.context_setting}. Must be 'full-text' for full-text prompts."
                )

        if "rag" in self.prompt_selection:
            if self.context_setting is None:
                raise ValueError(
                    f"Invalid context_setting: {self.context_setting}. Must be an integer or 'gold' for RAG prompts."
                )
            if (
                not isinstance(self.context_setting, int)
                and self.context_setting != "gold"
            ):
                raise ValueError(
                    f"Invalid context_setting: {self.context_setting}. Must be an integer or 'gold' for RAG prompts."
                )


def main(args: Args):

    # load the model
    kwargs = {}
    if args.model == "command-r-v01":
        model_path = "CohereForAI/c4ai-command-r-v01"
        max_model_len = (
            56_000  # do not set this to 128k to make KV cache fit into memory
        )
        gpu_memory_utilization = 1
        kwargs = {
            "enforce_eager": True,
            "swap_space": 0,
            "tensor_parallel_size": 2,  # tested on 2xA100
            "distributed_executor_backend": "mp",
        }
    elif args.model == "llama-8B-instruct-32k":
        model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        gpu_memory_utilization = 0.9
        max_model_len = 8192 * 4

    elif args.model == "llama-8B-instruct":
        model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        gpu_memory_utilization = 0.9
        max_model_len = 8192
    elif args.model == "mistral-7B-instruct-v02":
        model_path = "mistralai/Mistral-7B-Instruct-v0.2"
        gpu_memory_utilization = 0.9
        max_model_len = 8192 * 4
    else:
        raise ValueError(args.model)

    llm = LLM(
        model=model_path,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        **kwargs,
    )
    logger.info(f"Loaded model {args.model}")

    # load the data
    qa_loader = QuestionLoader(args.qa_file)
    paper_loader = PaperLoader(args.papers_file)
    logger.info(f"Loaded data from {args.qa_file} and {args.papers_file}")

    # initialize the tokenizer and sampling parameters
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1,
        max_tokens=1024,
        stop_token_ids=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
    )

    # create prompts
    prompt_template = (
        PROMPTS[args.prompt_selection]
        if args.prompt_template is None
        else args.prompt_template
    )
    logger.info(f"Using prompt template: {prompt_template}")

    if "full-text" in args.prompt_selection:
        logger.info("Creating full-text prompts")
        inputs, ids = make_prompts_full_text(
            tokenizer=llm.get_tokenizer(),
            prompt_template=prompt_template,
            max_model_len=max_model_len,
            model=args.model,
            qa_loader=qa_loader,
            paper_loader=paper_loader,
            apply_vllm_chat_template=True,
        )
    elif "rag" in args.prompt_selection:

        # load and process the retrieval results (or gold paragraphs)
        if args.context_setting == "gold":
            retrieval_file = "out/qrels.paragraphs.json"
        elif isinstance(args.context_setting, int):
            retrieval_file = args.retrieval_file
        else:
            raise ValueError(args.context_setting)
        logger.info(f"Loading retrieval results from {retrieval_file}")
        with open(retrieval_file) as fh:
            run = json.load(fh)

        df_run = []
        for question_id, document_id_to_score in run.items():
            for document_id, score in document_id_to_score.items():
                df_run.append(
                    {
                        "question_id": question_id,
                        "document_id": document_id,
                        "score": score,
                    }
                )
        df_run = pd.DataFrame(df_run)

        logger.info("Creating RAG prompts")
        inputs, ids = make_prompts_rag(
            tokenizer=llm.get_tokenizer(),
            prompt_template=prompt_template,
            max_model_len=max_model_len,
            context_setting=args.context_setting,
            model=args.model,
            sort_by=args.sort_by,
            df_run=df_run,
            qa_loader=qa_loader,
            paper_loader=paper_loader,
            apply_vllm_chat_template=True,
        )
    else:
        raise ValueError(args.prompt_selection)

    # run model inference on the prompts
    exp = f"{args.model}-{max_model_len//1000}k-{args.prompt_selection}"
    if "rag" in args.prompt_selection:
        exp += f"-{args.context_setting}"
    logger.info(f"Running inference for {exp}")
    logger.info(f"Random input: {random.choice(inputs)}")
    outputs = llm.generate(inputs, sampling_params)
    generations = process_outputs(outputs, ids)

    out_file = f"{str(args.output_dir)}/generations-{exp}.jsonl"
    logger.info(f"Writing generations to {out_file}")
    df_generations = pd.DataFrame(generations)
    df_generations.to_json(out_file, orient="records", lines=True)


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
