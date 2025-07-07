from functools import partial
from typing import Literal

import pandas as pd
import tiktoken

from peerqa.data_loader import PaperLoader, QuestionLoader
from peerqa.prompts import SYSTEM_PROMPT


def truncate(tokenizer, s: str, max_length: int):
    if isinstance(tokenizer, tiktoken.Encoding):
        enc = tokenizer.encode(s)
        enc = enc[:max_length]
        s = tokenizer.decode(enc)
    else:
        s = tokenizer.encode(s, max_length=max_length, truncation=True)
        s = tokenizer.decode(s, skip_special_tokens=True)
    return s


def inputs_from_prompts(
    prompts: list[str], tokenizer, model, apply_vllm_chat_template: bool = True
):
    inputs = []
    for p in prompts:
        # models with system prompt
        if model in [
            "command-r-v01",
            "llama-8B-instruct-32k",
            "llama-8B-instruct",
            "gpt-35-turbo-0613-16k",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "llama-3.3-70b-it-128k",
        ]:
            conversation = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {"role": "user", "content": p},
            ]
        # models without system prompt
        elif model in [
            "mistral-7B-instruct-v02",
            "deepseek-r1-llama-8b-128k",
            "deepseek-r1-qwen-7b-128k",
            "deepseek-r1-qwen-14b-128k",
            "deepseek-r1-qwen-32b-128k",
            "deepseek-r1-qwen3-8b-128k",
            "deepseek-r1-llama-70b-128k",
            "gemma-3-27b-it-128k",
        ]:
            conversation = [
                {
                    "role": "user",
                    "content": SYSTEM_PROMPT + " " + p,
                }
            ]
        else:
            raise ValueError(model)
        if apply_vllm_chat_template:
            conversation = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        inputs.append(conversation)
    return inputs


def get_top_N(df, N, sort_by: Literal["score", "document"] = "score"):
    if N == "gold":
        # use all gold paragraphs, so we just set N to a large number and sort by document order
        N = 1000
        sort_by = "document"
    if sort_by == "score":
        # sort the paragraphs/sentences by score first, break ties with document order
        return df.sort_values(
            by=["score", "document_id"], ascending=[False, True]
        ).head(N)
    elif sort_by == "document":
        # sort the paragraphs/sentences by document order
        return (
            df.sort_values(by=["score", "document_id"], ascending=[False, True])
            .head(N)
            .sort_values(by=["document_id"])
        )
    else:
        raise ValueError(sort_by)


def make_prompts_full_text(
    tokenizer,
    prompt_template: str,
    max_model_len: int,
    model: str,
    qa_loader,
    paper_loader,
    apply_vllm_chat_template: bool = True,
):
    prompts, ids = [], []

    for paper_id, document_ids, documents in paper_loader(granularity="paragraphs"):
        question_ids, questions, answerable = qa_loader.questions_un_answerable(
            paper_id=paper_id
        )

        if len(questions) == 0:
            continue

        paper = "\n".join(documents)
        paper = truncate(tokenizer, paper, max_model_len - 128)

        for question_id, question, question_answerable in zip(
            question_ids, questions, answerable
        ):
            prompt = prompt_template.format(paper=paper, question=question)
            prompts.append(prompt)
            ids.append(
                {
                    "paper_id": paper_id,
                    "question_id": question_id,
                    "question": question,
                    "answerable": question_answerable,
                }
            )

    inputs = inputs_from_prompts(prompts, tokenizer, model, apply_vllm_chat_template)

    return inputs, ids


def make_prompts_rag(
    tokenizer,
    prompt_template: str,
    max_model_len: int,
    context_setting,
    model: str,
    df_run: pd.DataFrame,
    qa_loader: QuestionLoader,
    paper_loader: PaperLoader,
    sort_by: str = "score",
    apply_vllm_chat_template: bool = True,
    join_with: Literal["linebreak", "references"] = "linebreak",
):

    # top_N_paragraphs = get_top_N(df_run, N=context_setting, sort_by=sort_by)
    if sort_by == "score":
        fn = partial(get_top_N, sort_by="score")
    elif sort_by == "document":
        fn = partial(get_top_N, sort_by="document")
    else:
        raise ValueError(sort_by)

    # Group by question_id and apply the function to get the top N paragraphs
    top_N_paragraphs = df_run.groupby("question_id").apply(fn, N=context_setting)
    top_N_paragraphs.reset_index(drop=True, inplace=True)
    
    prompts, ids = [], []
    for paper_id, document_ids, documents in paper_loader(granularity="paragraphs"):
        question_ids, questions, answerable = qa_loader.questions_un_answerable(
            paper_id=paper_id
        )

        if len(questions) == 0:
            continue

        for question_id, question, question_answerable in zip(
            question_ids, questions, answerable
        ):
            rag_documents = []
            top_document_ids = top_N_paragraphs[
                top_N_paragraphs.question_id == question_id
            ].document_id.values.astype(int)
            for top_document_id in top_document_ids:
                top_document = documents[document_ids.index(top_document_id)]
                rag_documents.append(top_document)

            if len(rag_documents) == 0:
                continue

            # paragraphs = "\n\n".join(rag_documents)
            if join_with == "linebreak":
                paragraphs = "\n\n".join(rag_documents)
                offset = 128
            elif join_with == "references":
                paragraphs = ""
                for doc_id, doc in zip(document_ids, rag_documents):
                    paragraphs += f"[{doc_id}] {doc}\n"
                offset = 384  # have a longer offset for references to make sure the references are included
            else:
                raise ValueError(join_with)
            paragraphs = truncate(tokenizer, paragraphs, max_model_len - offset)

            prompt = prompt_template.format(question=question, paragraphs=paragraphs)
            prompts.append(prompt)
            ids.append(
                {
                    "paper_id": paper_id,
                    "question_id": question_id,
                    "document_ids": top_document_ids,
                    "question": question,
                    "answerable": question_answerable,
                    "prompt": prompt,
                }
            )

    inputs = inputs_from_prompts(prompts, tokenizer, model, apply_vllm_chat_template)

    return inputs, ids


def process_outputs(outputs, ids):
    generations = []
    for _id, output in zip(ids, outputs):
        kwargs = {}
        generation = output.outputs[0].text
        # check if any reasoning tokens are in the generation
        if any(reasoning_token in generation for reasoning_token in ["</think>"]):

            # check if deepseek-r1 reasoning tokens are in the generation
            if any(reasoning_token in generation for reasoning_token in ["</think>"]):
                # NOTE: this removes any text before the first <think> token
                to_think = generation.index("</think>") + len("</think>")
                reasoning = generation[: to_think]
                generation = generation[to_think:].strip("\n")
                kwargs = {"reasoning": reasoning}
        generations.append({**_id, "generation": generation, **kwargs})
    return generations
