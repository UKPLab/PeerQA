import logging
from logging.config import fileConfig
from typing import List, Union

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


class DenseWrapper:
    "Wrapper providing base functions of `SentenceTransformers` with non-native model."

    def encode(*args, **kwargs):
        raise NotImplementedError()


class HFBase(DenseWrapper):
    def __init__(self, model_name: str, pooling: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.pooling = pooling

        if torch.cuda.is_available():
            logger.info("Loading Model to GPU.")
            self.model.to("cuda")
        self.model.eval()

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
    ):
        embeddings = []
        for batch_sentences in tqdm(
            self._batchify(sentences, batch_size),
            disable=not show_progress_bar,
            ncols=80,
        ):
            inputs = self.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False,
                max_length=512,
            )
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")

            with torch.no_grad():
                output = self.model(**inputs)

            if self.pooling == "first_token":
                # take the first token ([CLS]) in the batch as the embedding
                batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
            elif self.pooling == "mean_pooling":
                batch_embeddings = (
                    self._mean_pooling(output[0], inputs["attention_mask"])
                    .cpu()
                    .numpy()
                )
            elif self.pooling == "splade":
                logits = output.logits
                batch_embeddings = (
                    torch.max(
                        torch.log(1 + torch.relu(logits))
                        * inputs["attention_mask"].unsqueeze(-1),
                        dim=1,
                    )[0]
                    .cpu()
                    .numpy()
                )
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}.")
            embeddings.append(batch_embeddings)

        embeddings = np.concatenate(embeddings)

        assert len(embeddings) == len(
            sentences
        ), f"{len(embeddings)=} != {len(sentences)=}"

        return embeddings

    @staticmethod
    def _batchify(x: list, batch_size: int):
        num_batches = len(x) // batch_size
        if len(x) % batch_size:
            num_batches += 1
        for i in range(num_batches):
            yield x[i * batch_size : (i + 1) * batch_size]

    @staticmethod
    def _mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings


class Splade(HFBase):
    def __init__(self, model_name: str, pooling: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.pooling = pooling

        if torch.cuda.is_available():
            logger.info("Loading Model to GPU.")
            self.model.to("cuda")
        self.model.eval()
