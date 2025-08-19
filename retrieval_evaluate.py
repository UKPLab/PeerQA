import base64
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from logging.config import fileConfig
from pathlib import Path
from typing import Literal

import numpy as np
import simple_parsing
from pytrec_eval import RelevanceEvaluator
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from peerqa.utils import url_save_hash, url_save_str

fileConfig("logging.ini")
logger = logging.getLogger(__name__)

from pytrec_eval import RelevanceEvaluator

at_k = [1, 5, 10, 20, 50, 100, 1000]
map = "map_cut." + ",".join([str(k) for k in at_k])
recall = "recall." + ",".join([str(k) for k in at_k])
precision = "P." + ",".join([str(k) for k in at_k])
rprecision = "Rprec." + ",".join([str(k) for k in at_k])
mrr = "recip_rank." + ",".join([str(k) for k in at_k])
measures = {map, recall, precision, rprecision, mrr}


@dataclass
class Args:
    query_model: str
    output_dir: Path = field(default=Path("out"))
    sim_fn: Literal["cos", "dot", "sparse", "rr", "cross", "maxsim"] = "dot"
    granularity: Literal["paragraphs", "sentences"] = "paragraphs"
    template: str = None
    hyde_file: Path = None


def main(args: Args):

    with open(args.output_dir / f"qrels.{args.granularity}.json", "r") as f:
        qrels = json.load(f)

    model_str = url_save_str(args.query_model)
    if args.use_hyde:
        model_str += "-hyde"
    run_file = args.output_dir / (
        f"run-{args.granularity}-{model_str}-{args.sim_fn}"
        + (f"-{url_save_hash(args.template)}" if args.template is not None else "")
        + ".json"
    )

    with open(run_file, "r") as f:
        run = json.load(f)

    relevance_evaluator = RelevanceEvaluator(qrels, measures=measures)
    question_id_to_metrics = relevance_evaluator.evaluate(run)

    metrics_flat = defaultdict(list)
    for question_id, metrics in question_id_to_metrics.items():
        for metric, value in metrics.items():
            metrics_flat[metric].append(value)

    metrics = defaultdict(dict)
    for metric, values in metrics_flat.items():
        metrics[metric]["mean"] = np.mean(values)
        metrics[metric]["std"] = np.std(values)

    for metric, values in metrics.items():
        logger.info(f"{metric:12s}: {values['mean']:.4f} +- {values['std']:.4f}")

    # Replace "run" with "metrics"
    filename = run_file.stem
    new_filename = "metrics" + filename[3:] + run_file.suffix
    with open(
        run_file.with_name(new_filename),
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    args, _ = simple_parsing.parse_known_args(Args)
    args.use_hyde = args.hyde_file is not None
    with logging_redirect_tqdm():
        logger.info(args)
        main(args)
