import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import Literal

import simple_parsing
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from peerqa.utils import url_save_hash

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    output_dir: Path = field(default=Path("out"))
    granularity: Literal["sentences", "paragraphs"] = "sentences"
    template: str = None
    override: bool = True


def main(args):

    if args.template is not None:
        template_hash = url_save_hash(args.template)
        out_path_suffix = f"-{template_hash}"
    else:
        out_path_suffix = ""
    out_path = (
        args.output_dir / f"run-{args.granularity}-colbert-maxsim{out_path_suffix}.json"
    )
    if (not args.override) and out_path.exists():
        logger.info(f"Skipping since {out_path=} already exists.")
        return
    logger.info(f"Will write results to {out_path=}.")

    subdir = f"colbert-{args.granularity}"
    if args.template is not None:
        template_hash = url_save_hash(args.template)
        logger.info(f"Adding template hash {template_hash} to subdir.")
        subdir += f"-{template_hash}"
    # experiment_dir = str(args.output_dir / subdir)
    experiment_dir = args.output_dir / subdir

    # ranking_files = list(
    #     (
    #         Path(experiment_dir).glob(
    #             "*/peer_qa_experiments.search_colbert/*/*/*/paper.nbits=2.ranking.tsv"
    #         )
    #     )
    # )
    ranking_files = list(experiment_dir.glob("**/paper.nbits=2.ranking.tsv"))
    run = defaultdict(dict)
    for ranking_file in tqdm(ranking_files, ncols=80):
        with open(ranking_file, "r") as fh:
            for qid, doc_id, rank, score in csv.reader(fh, delimiter="\t"):
                run[qid][doc_id] = float(score)

    with open(out_path, "w") as f:
        json.dump(run, f, indent=2)


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
