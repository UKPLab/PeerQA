# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "colbert-ai",
#     "faiss-cpu",
#     "numpy==1.24.1",
#     "sentence-transformers==2.7.0",
#     "setuptools",
#     "simple-parsing",
#     "torch==1.13.1",
#     "transformers==4.40.0",
# ]
#
# [tool.uv.sources]
# colbert-ai = { git = "https://github.com/timbmg/ColBERT" }
# ///
import logging
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import Literal

import simple_parsing
from colbert import Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from peerqa.utils import url_save_hash, url_save_str

fileConfig("logging.ini")
logger = logging.getLogger(__name__)


@dataclass
class Args:
    output_dir: Path = field(default=Path("out"))
    granularity: Literal["sentences", "paragraphs"] = "sentences"
    template: str = None


def main(args):

    subdir = f"colbert-{args.granularity}"
    if args.template is not None:
        template_hash = url_save_hash(args.template)
        logger.info(f"Adding template hash {template_hash} to subdir.")
        subdir += f"-{template_hash}"
    experiment_dir = str(args.output_dir / subdir)

    query_files = list((args.output_dir / subdir).glob("*/queries.tsv"))
    for query_file in tqdm(query_files, ncols=80):
        paper_id = str(query_file.parts[-2])

        index_path = Path(experiment_dir) / paper_id / "indexes" / "paper.nbits=2" / "ivf.pid.pt"
        assert index_path.exists(), index_path

        with Run().context(
            RunConfig(nranks=1, root=experiment_dir, experiment=paper_id)
        ):

            config = ColBERTConfig(
                root=experiment_dir,
            )
            searcher = Searcher(index="paper.nbits=2", config=config)
            queries = Queries(str(query_file))
            ranking = searcher.search_all(queries, k=10000)
            ranking.save("paper.nbits=2.ranking.tsv")


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
