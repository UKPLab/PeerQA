import os

import pandas as pd
import spacy

os.environ["GROBID_HOST"] = os.environ.get("GROBID_HOST", "http://localhost:8070")

import json
import logging
import shutil
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path

import simple_parsing
from intertext_graph.itsentsplitter import IntertextSentenceSplitter
from nlpeer.data.create.parse import pdf_to_tei, tei_to_itg
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from peerqa.sentenizer import SPECIAL_SPLIT_TOKEN, SentenizerPipeline

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    nlpeer_path: Path
    data_path: Path = field(default=Path("data"))
    qa_file: Path = field(default=Path("data/qa.jsonl"))
    override: bool = False


SENTENIZIER_PIPELINE = SentenizerPipeline(["special_token", "punct", "enum", "empty"])


def flatten_paper_itg_json(itg: dict) -> list[dict]:
    """Flatten an .itg.json to a list of sentences with their position in the paper."""
    paper_text = []
    for sentence in filter(lambda node: node["ntype"] == "s", itg["span_nodes"]):
        text = sentence["content"]
        paper_text.append(text)

    text = f"{SPECIAL_SPLIT_TOKEN}".join(paper_text)
    sentences = SENTENIZIER_PIPELINE(text)
    flat = [{"pos": pos, "text": text} for pos, text in enumerate(sentences)]

    return flat


def find_in_list_of_dict(l, k, v):
    """Find all dictionaries in a list of dictionaries where a key has a specific value."""
    return list(filter(lambda li: li[k] == v, l))


nlp = spacy.load("en_core_sci_sm")


def sentinize(s: str):
    """Split a string into sentences using spacy."""
    return [str(i) for i in nlp(s).sents]


def main(args: Args):

    # get all pdfs in the data path
    pdfs_to_process = list(args.data_path.glob("**/paper.pdf"))

    # get the nlpeer papers and copy the camera-ready version to the data path
    qa_data = pd.read_json(args.qa_file, lines=True)
    nlpeer_papers = (
        qa_data[qa_data.paper_id.str.startswith("nlpeer")].paper_id.unique().tolist()
    )
    for paper_id in nlpeer_papers:
        _, venue, nlpeer_id = paper_id.split("/")
        # get all dirs starting with v* (e.g. v1, v2, ...)
        paper_path = args.nlpeer_path / venue / "data" / nlpeer_id
        paper_versions = sorted(paper_path.glob("v*"))
        # get the latest version
        paper_file_nlpeer = paper_versions[-1] / "paper.pdf"
        # copy the camera-ready version to the data path
        paper_file_peerqa = args.data_path / "nlpeer" / venue / nlpeer_id / "paper.pdf"
        paper_file_peerqa.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(paper_file_nlpeer, paper_file_peerqa)

        pdfs_to_process.append(paper_file_peerqa)

    for paper_pdf_file in tqdm(pdfs_to_process, ncols=80, desc="Processing PDFs"):
        base_path = paper_pdf_file.parent

        # use GROBID to convert the PDF to TEI
        tei_path = base_path / "paper.tei.xml"
        if args.override or not tei_path.exists():
            logger.debug(f"Processing {paper_pdf_file}")
            status, tei = pdf_to_tei(str(paper_pdf_file))
            with open(tei_path, "w") as f:
                f.write(tei)

        # convert the TEI to an .itg.json
        itg_path = base_path / "paper.itg.json"
        if args.override or not itg_path.exists():
            try:
                itg = tei_to_itg(str(tei_path))
            except:
                logger.error(f"Error: {tei_path}")
                continue

            itg = IntertextSentenceSplitter(itg).add_sentences_to_itg()

            # dump itg to file
            with open(itg_path, "w") as f:
                f.write(itg.to_json())

        # convert the .itg.json to a list of sentences including the paragraph and sentence index
        content_path = base_path / "paper.content.jsonl"
        if args.override or not content_path.exists():
            with open(itg_path) as fp:
                itg = json.load(fp)

            sentences = list(filter(lambda n: n["ntype"] == "s", itg["span_nodes"]))

            content = []
            last_pidx = 0
            figures_tables_processed = False
            for node in itg["nodes"]:
                if node["ntype"] == "title":
                    assert not figures_tables_processed
                    paragraph_ix = node["ix"]
                    pidx = int(paragraph_ix.split("_")[1])
                    last_pidx = pidx
                    content.append(
                        {
                            "idx": len(content),
                            "pidx": pidx,
                            "sidx": 0,
                            "type": "title",
                            "content": node["content"].title(),
                        }
                    )
                elif node["ntype"] in ["abstract", "heading"]:
                    assert not figures_tables_processed
                    paragraph_ix = node["ix"]
                    pidx = int(paragraph_ix.split("_")[1])
                    last_pidx = pidx
                    content.append(
                        {
                            "idx": len(content),
                            "pidx": pidx,
                            "sidx": 0,
                            "type": "heading",
                            "content": node["content"].title(),
                        }
                    )
                    last_heading = node["content"]
                elif node["ntype"] == "p":
                    assert not figures_tables_processed
                    paragraph_ix = node["ix"]
                    pidx = int(paragraph_ix.split("_")[1])
                    last_pidx = pidx
                    # find all sentences beloging to the paragraph
                    paragraph_sentences = find_in_list_of_dict(
                        sentences, "src_ix", paragraph_ix
                    )
                    assert paragraph_sentences, f"{itg_path=} {node=}"
                    for paragraph_sentence in paragraph_sentences:
                        content.append(
                            {
                                "idx": len(content),
                                "pidx": pidx,
                                "sidx": int(paragraph_sentence["ix"].split("@")[1]),
                                "type": "sentence",
                                "content": paragraph_sentence["content"],
                                "last_heading": last_heading,
                            }
                        )
                elif node["ntype"] == "list_item":
                    assert not figures_tables_processed

                    paragraph_ix = node["ix"]
                    pidx = int(paragraph_ix.split("_")[1])
                    last_pidx = pidx

                    list_item_sents = sentinize(node["content"])

                    for sidx, sent in enumerate(list_item_sents):
                        content.append(
                            {
                                "idx": len(content),
                                "pidx": pidx,
                                "sidx": sidx,
                                "type": "list_item",
                                "content": sent,
                                "last_heading": last_heading,
                            }
                        )
                elif node["ntype"] == "formula":
                    paragraph_ix = node["ix"]
                    content.append(
                        {
                            "idx": len(content),
                            "pidx": int(paragraph_ix.split("_")[1]),
                            "sidx": 0,
                            "type": "formula",
                            "content": node["content"],
                            "last_heading": last_heading,
                        }
                    )
                elif node["ntype"] in ["figure", "table"]:
                    figures_tables_processed = True
                    caption = node["meta"]["caption"]
                    if caption is None:
                        continue
                    caption_sents = sentinize(caption)
                    last_pidx += 1
                    for sidx, sent in enumerate(caption_sents):
                        content.append(
                            {
                                "idx": len(content),
                                "pidx": last_pidx,
                                "sidx": sidx,
                                "type": node["ntype"],
                                "content": sent,
                            }
                        )

                with open(content_path, "w", encoding="utf-8") as file:
                    pd.DataFrame(content).to_json(
                        file, lines=True, force_ascii=False, orient="records"
                    )

    df = []
    for paper_content_file in args.data_path.glob("**/paper.content.jsonl"):
        _df = pd.read_json(paper_content_file, lines=True)
        # strip the args.data_dir and file name
        _df["paper_id"] = str(Path(*paper_content_file.parts[1:]).parent)
        df.append(_df)
    df = pd.concat(df)
    df.to_json(
        args.data_path / "papers.jsonl", lines=True, force_ascii=False, orient="records"
    )


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
