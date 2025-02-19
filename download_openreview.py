import logging
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path

import pandas as pd
import PyPDF2
import requests
import simple_parsing
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    data_path: Path = field(default=Path("data"))
    qa_file: Path = field(default=Path("data/qa.jsonl"))
    override: bool = False
    seconds_between_requests: int = 1
    delete_supplementary: bool = True


def maybe_delete_supplementary(supplement_dir: Path, delete_supplementary: bool):
    if delete_supplementary:
        logger.debug(
            f"Deleting supplementary material for {supplement_dir.parent.name}"
        )
        shutil.rmtree(supplement_dir)


def main(args: Args):

    qa_data = pd.read_json(args.qa_file, lines=True)

    # filter out papers that are not from OpenReview
    qa_data = qa_data[qa_data.paper_id.str.startswith("openreview")]

    # the paper_ids look like "openreview/conference/forum_id"
    # we only need the forum_id to download the PDFs
    conference_forum_ids = [
        paper_id.split("/")[1:] for paper_id in qa_data.paper_id.unique()
    ]

    for conference, forum_id in tqdm(
        conference_forum_ids, ncols=80, desc="Downloading PDFs"
    ):

        # create the output path
        output_dir = args.data_path / "openreview" / conference / forum_id
        paper_file = output_dir / "paper.pdf"
        if not paper_file.exists() or args.override:
            output_dir.mkdir(parents=True, exist_ok=True)
            # download the PDF
            openreview_url = f"https://openreview.net/pdf?id={forum_id}"
            logger.debug(f"Downloading paper {forum_id}")
            r = requests.get(openreview_url, stream=True)
            with open(paper_file, "wb") as f:
                f.write(r.content)
            # wait a bit before the next request
            time.sleep(args.seconds_between_requests)

        # download the supplementary material to get the appendix
        if conference == "NeurIPS-2022-dabt":
            if forum_id in ["dh_MkX0QfrK", "dwi57JI_-K"]:
                continue

            logger.debug(f"Downloading supplementary material for {forum_id}")
            supplement_dir = output_dir / "supplementary"
            supplementary_material_url = f"https://openreview.net/attachment?id={forum_id}&name=supplementary_material"
            r = requests.get(supplementary_material_url, stream=True)
            file_type = r.headers.get("Content-Type").split("/")[-1]
            if file_type == "pdf":
                supplementary_file = supplement_dir / "supplementary.pdf"
            elif file_type == "zip":
                supplementary_file = supplement_dir / "supplementary.zip"
            elif file_type == "html":
                logger.debug(
                    f"Skipping {forum_id} because of no supplementary material."
                )
                continue
            else:
                raise ValueError(f"Unknown file type {file_type}")

            supplement_dir.mkdir(parents=True, exist_ok=True)

            if not supplementary_file.exists() or args.override:
                with open(supplementary_file, "wb") as f:
                    f.write(r.content)

                if file_type == "zip":
                    with zipfile.ZipFile(supplementary_file, "r") as zip_ref:
                        zip_ref.extractall(supplement_dir)

            if (output_dir / "paper_original.pdf").exists() and not args.override:
                # skip if the paper has already been processed
                logger.debug(
                    f"Skipping appendix merging for {forum_id} because the paper has already been processed."
                )
                maybe_delete_supplementary(supplement_dir, args.delete_supplementary)
                continue

            if (
                not supplement_dir.exists()
                and not (output_dir / "supplementary.pdf").exists()
            ):
                # if there is no supplementary material, skip
                logger.debug(
                    f"Skipping appendix merging for {forum_id} because there is no supplementary material."
                )
                maybe_delete_supplementary(supplement_dir, args.delete_supplementary)
                continue

            supplementary_pdf_files = list(
                filter(
                    lambda p: not p.name.startswith("."),
                    supplement_dir.glob("**/*.pdf"),
                )
            )
            if not len(supplementary_pdf_files) <= 1:

                # check if there is an appendix
                triggers = ["appendi", "appdx", "supp", "supmat"]
                appendix_pdf_files = [
                    f
                    for f in supplementary_pdf_files
                    if any(t in f.name.lower() for t in triggers)
                ]

                if not len(appendix_pdf_files) == 1:

                    triggers = ["paper", "main"]
                    appendix_pdf_files_filtered = [
                        f
                        for f in appendix_pdf_files
                        if any(t not in f.name.lower() for t in triggers)
                    ]
                    if len(appendix_pdf_files_filtered) == 1:
                        supplementary_pdf_files = appendix_pdf_files_filtered
                    else:
                        logger.debug(f"Could not find appendix for {forum_id}")
                        maybe_delete_supplementary(
                            supplement_dir, args.delete_supplementary
                        )
                        continue

                supplementary_pdf_files = appendix_pdf_files

            files_to_merge = [paper_file] + supplementary_pdf_files
            if len(files_to_merge) <= 1:
                logger.debug(
                    f"Skipping appendix merging for {forum_id} because no appendix files were found."
                )
                maybe_delete_supplementary(supplement_dir, args.delete_supplementary)
                continue

            logger.debug(f"Merging {files_to_merge}")
            merger = PyPDF2.PdfWriter()
            for pdf in files_to_merge:
                try:
                    merger.append(pdf)
                except Exception as e:
                    logger.error(f"Error with {pdf}")
                    raise e
            # rename the original paper to paper_original.pdf
            shutil.move(paper_file, output_dir / "paper_original.pdf")
            # write the merged PDF to paper.pdf
            merger.write(paper_file)
            merger.close()

            maybe_delete_supplementary(supplement_dir, args.delete_supplementary)


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
