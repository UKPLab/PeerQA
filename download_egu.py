import logging
import time
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path

import pandas as pd
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
    seconds_between_requests: int = 1


def main(args: Args):

    qa_data = pd.read_json(args.qa_file, lines=True)

    # filter out papers that are not from OpenReview
    qa_data = qa_data[qa_data.paper_id.str.startswith("egu")]
    paper_ids = qa_data.paper_id.unique()

    for paper_id in tqdm(paper_ids, ncols=80, desc="Downloading PDFs"):
        _, journal, paper_id = paper_id.split("/")
        volume, page, year = paper_id.split("-")

        # create the output path
        output_path_pdf = args.data_path / "egu" / journal / paper_id / "paper.pdf"
        output_path_pdf.parent.mkdir(parents=True, exist_ok=True)
        # download the PDF
        egu_url = f"https://{journal}.copernicus.org/articles/{volume}/{page}/{year}/{journal}-{paper_id}.pdf"
        logger.debug(f"Downloading {egu_url}")
        r = requests.get(egu_url)
        with open(output_path_pdf, "wb") as f:
            f.write(r.content)
        # wait a bit before the next request
        time.sleep(args.seconds_between_requests)


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
