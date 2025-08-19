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
import torch
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
        "deepseek-r1-llama-8b-128k",
        "deepseek-r1-qwen-7b-128k",
        "deepseek-r1-qwen-14b-128k",
        "deepseek-r1-qwen-32b-128k",
        "deepseek-r1-qwen3-8b-128k",
        "deepseek-r1-llama-70b-128k",
        "gemma-3-27b-it-128k",
        "llama-3.3-70b-it-128k",
    ] = "llama-8B-instruct"
    context_setting: int | str = None
    vllm_bs: int = 0

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
    dtype = "float16"  # default dtype for vLLM
    if args.model == "command-r-v01":
        model_path = "CohereForAI/c4ai-command-r-v01"
        max_model_len = (
            56_000  # do not set this to 128k to make KV cache fit into memory
        )
        gpu_memory_utilization = 1
        kwargs = {
            "tensor_parallel_size": 2,
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
    elif args.model == "deepseek-r1-llama-8b-128k":
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        gpu_memory_utilization = 0.9
        max_model_len = 131072
    elif args.model == "deepseek-r1-qwen-7b-128k":
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        gpu_memory_utilization = 0.9
        max_model_len = 131072
    elif args.model == "deepseek-r1-qwen3-8b-128k":
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        gpu_memory_utilization = 0.9
        max_model_len = 131072
    elif args.model == "deepseek-r1-qwen-14b-128k":
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        gpu_memory_utilization = 0.9
        max_model_len = 131072
    elif args.model == "deepseek-r1-qwen-32b-128k":
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        max_model_len = 131072 // 2 # has to be reduced to fit KV Cache
        gpu_memory_utilization = 0.98
        kwargs = {
            "tensor_parallel_size": 2,  # tested on 2xA100
            "distributed_executor_backend": "mp",
        }
    elif args.model == "deepseek-r1-llama-70b-128k":
        model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-70B"
        max_model_len = 131072 // 2 # has to be reduced to fit KV Cache
        gpu_memory_utilization = 0.95
        kwargs = {
            "tensor_parallel_size": torch.cuda.device_count(),
            "distributed_executor_backend": "mp",
        }
    elif args.model == "gemma-3-27b-it-128k":
        model_path = "google/gemma-3-27b-it"
        max_model_len = 131072 // 2 # has to be reduced to fit KV Cache
        gpu_memory_utilization = 0.9
        dtype="bfloat16" # Value error, The model type 'gemma3' does not support float16. Reason: Numerical instability. Please use bfloat16 or float32 instead.
        kwargs = {
            "tensor_parallel_size": torch.cuda.device_count(),
            "distributed_executor_backend": "mp",
        }
    elif args.model == "llama-3.3-70b-it-128k":
        model_path = "meta-llama/Llama-3.3-70B-Instruct"
        max_model_len = 131072 // 2 # has to be reduced to fit KV Cache
        gpu_memory_utilization = 0.95
        kwargs = {
            "tensor_parallel_size": torch.cuda.device_count(),
            "distributed_executor_backend": "mp",
        }
    else:
        raise ValueError(args.model)

    llm = LLM(
        model=model_path,
        dtype=dtype,
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
    
    # occasionally, the model runs out of memory when generating all inputs at once
    # in this case, we can split the inputs into batches by setting vllm_bs > 0
    if args.vllm_bs > 0:
        generations = []
        num_batches = len(inputs) // args.vllm_bs
        if len(inputs) % args.vllm_bs > 0:
            num_batches += 1
        for i in range(0, num_batches):
            logger.info(f"Batch {i+1}/{num_batches}")
            batch_inputs = inputs[i * args.vllm_bs : (i + 1) * args.vllm_bs]
            batch_ids = ids[i * args.vllm_bs : (i + 1) * args.vllm_bs]
            outputs = llm.generate(batch_inputs, sampling_params)
            _generations = process_outputs(outputs, batch_ids)
            generations.extend(_generations)
    else:
        outputs = llm.generate(inputs, sampling_params)
        generations = process_outputs(outputs, ids)

    out_file = str(args.output_dir / f"generations-{exp}.jsonl")
    logger.info(f"Writing generations to {out_file}")
    df_generations = pd.DataFrame(generations)
    df_generations.to_json(out_file, orient="records", lines=True)

    logger.debug(f"Sample Generation:\n{df_generations.sample(1).to_json(indent=2)}")


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
