import json
import logging
import time
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import Literal

import pandas as pd
import simple_parsing
import tiktoken
from openai import OpenAI
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from peerqa.data_loader import PaperLoader, QuestionLoader
from peerqa.generate_utils import make_prompts_full_text, make_prompts_rag
from peerqa.prompts import PROMPTS

fileConfig("logging.ini")
logger = logging.getLogger(__name__)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)


@dataclass
class Args:

    openai_api_key: str
    model: Literal[
        "gpt-35-turbo-0613-16k",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
    ] = "gpt-4o-2024-08-06"
    estimate_only: bool = False
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
    ] = "rag"
    prompt_template: str = None
    sort_by: Literal["score", "document"] = "score"
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


MODEL_TOKEN_COST = {
    "gpt-35-turbo-0613-16k": {
        "input_token": 0.003 / 1_000,
        "output_token": 0.004 / 1_000,
    },
    "gpt-4o-2024-08-06": {
        "input_token": 0.0025 / 1_000,
        "output_token": 0.01 / 1_000,
    },
    "gpt-4o-mini-2024-07-18": {
        "input_token": 0.00015 / 1_000,
        "output_token": 0.0006 / 1_000,
    },
}


def openai_generate(client, model, inputs, ids):
    def complete(messages):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            top_p=1,
        )
        return completion

    total_cost = 0
    completions = []
    with tqdm(total=len(inputs)) as pbar:
        for messages in inputs:

            retries = 0
            while True:
                try:
                    completion = complete(messages)
                    break
                except Exception as e:
                    logger.error(e)
                    retries += 1
                    logger.info(f"Retrying... {retries}")
                    time.sleep(3 * retries)
            retries = 0
            completions.append(completion)

            # calculate cost
            input_tokens = completion.usage.prompt_tokens
            output_tokens = completion.usage.completion_tokens
            cost = (
                input_tokens * MODEL_TOKEN_COST[model]["input_token"]
                + output_tokens * MODEL_TOKEN_COST[model]["output_token"]
            )
            total_cost += cost
            pbar.update(1)
            pbar.set_description(desc=f"Cost: {total_cost:.4f}$")
            time.sleep(0.5)

    return completions


def main(args: Args):

    # load the data
    qa_loader = QuestionLoader(args.qa_file)
    paper_loader = PaperLoader(args.papers_file)
    logger.info(f"Loaded data from {args.qa_file} and {args.papers_file}")

    if args.model == "gpt-35-turbo-0613-16k":
        max_model_len = 16385
    elif args.model == "gpt-4o-2024-08-06":
        max_model_len = 128_000
    elif args.model == "gpt-4o-mini-2024-07-18":
        max_model_len = 128_000
    else:
        raise ValueError(args.model)

    # create prompts
    prompt_template = (
        PROMPTS[args.prompt_selection]
        if args.prompt_template is None
        else args.prompt_template
    )
    logger.info(f"Using prompt template: {prompt_template}")

    tokenizer = tiktoken.encoding_for_model(args.model)
    if "full-text" in args.prompt_selection:
        logger.info("Creating full-text prompts")
        inputs, ids = make_prompts_full_text(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_model_len=max_model_len,
            model=args.model,
            qa_loader=qa_loader,
            paper_loader=paper_loader,
            apply_vllm_chat_template=False,
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
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_model_len=max_model_len,
            context_setting=args.context_setting,
            model=args.model,
            sort_by=args.sort_by,
            df_run=df_run,
            qa_loader=qa_loader,
            paper_loader=paper_loader,
            apply_vllm_chat_template=False,
        )
    else:
        raise ValueError(args.prompt_selection)

    # run model inference on the prompts
    exp = f"{args.model}-{max_model_len//1000}k-{args.prompt_selection}"
    if "rag" in args.prompt_selection:
        exp += f"-{args.context_setting}"

    if "full-text" in args.prompt_selection:
        logger.info("Creating full-text prompts")
        inputs, ids = make_prompts_full_text(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_model_len=max_model_len,
            model=args.model,
            qa_loader=qa_loader,
            paper_loader=paper_loader,
            apply_vllm_chat_template=False,
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
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_model_len=max_model_len,
            context_setting=args.context_setting,
            model=args.model,
            sort_by=args.sort_by,
            df_run=df_run,
            qa_loader=qa_loader,
            paper_loader=paper_loader,
            apply_vllm_chat_template=False,
        )
    else:
        raise ValueError(args.prompt_selection)

    # estimate costs
    input_tokens = 0
    for i in inputs:
        for message in i:
            input_tokens += len(tokenizer.encode(message["content"]))
    logger.info(f"Input tokens: {input_tokens}")
    estimated_average_output_tokens = 256
    estimated_cost = (
        input_tokens * MODEL_TOKEN_COST[args.model]["input_token"]
        + estimated_average_output_tokens * MODEL_TOKEN_COST[args.model]["output_token"]
    )
    logger.info(f"Estimated costs: {estimated_cost:.4f}$")
    if args.estimate_only:
        return

    client = OpenAI(
        api_key=args.openai_api_key,
    )

    completions = openai_generate(client, args.model, inputs, ids)
    generations = []
    for _id, completion in zip(ids, completions):
        generations.append({**_id, "generation": completion.choices[0].message.content})

    out_file = f"{str(args.output_dir)}/generations-{exp}.jsonl"
    logger.info(f"Writing generations to {out_file}")
    df_generations = pd.DataFrame(generations)
    df_generations.to_json(out_file, orient="records", lines=True)


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
