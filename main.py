"""Run GMPCR with the selected configuration for each supported dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gmpcr import (
    Gmpcr,
    configure_eval_metrics,
    set_random_seed,
)


ROOT = Path(__file__).resolve().parent
BEST_CONFIG_FILES = {
    "3dmatch": ROOT / "configs" / "3dmatch.json",
    "3dLomatch": ROOT / "configs" / "3dlomatch.json",
    "scannet": ROOT / "configs" / "scannet.json",
    "ETH": ROOT / "configs" / "eth.json",
}

REQUIRED_CONFIG_KEYS = {
    "seed",
    "comp_threshold",
    "fcg_threshold",
    "num_sample",
    "top_overlap",
    "scg_keep_ratio",
    "topk",
    "inlier_threshold",
    "tau_2",
    "num_iterations",
    "max_cliques",
    "weight_iterations",
    "translation_weight_iterations",
    "lambda_min",
    "lambda_max",
    "kappa",
    "translation_kappa",
    "rho",
    "translation_rho",
    "w_min",
    "w_max",
    "residual_scale",
    "translation_residual_scale",
    "pair_confidence_weight_floor",
    "pair_confidence_weight_power",
}

def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GMPCR using the automatically selected configuration "
            "for 3DMatch, 3DLoMatch, ScanNet, or ETH."
        )
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=tuple(BEST_CONFIG_FILES),
        default="3dmatch",
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        default=str(ROOT / "data"),
        help="common parent directory of the four benchmark datasets",
    )
    parser.add_argument(
        "--save-dir",
        "--save_dir",
        default="./pre",
        help="directory used for predicted poses and evaluation logs",
    )
    parser.add_argument(
        "--scene",
        default=None,
        help="evaluate one scene; the default evaluates the complete dataset",
    )
    return parser


def load_selected_config(dataset):
    config_path = BEST_CONFIG_FILES[dataset]
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    if config.get("dataset") != dataset:
        raise ValueError(
            f"Dataset mismatch in {config_path}: expected {dataset}, "
            f"found {config.get('dataset')!r}"
        )

    allowed = REQUIRED_CONFIG_KEYS | {"dataset"}
    unknown = set(config) - allowed
    missing = REQUIRED_CONFIG_KEYS - set(config)
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown keys: {sorted(unknown)}")
        if missing:
            details.append(f"missing keys: {sorted(missing)}")
        raise ValueError(f"Invalid configuration in {config_path}: {'; '.join(details)}")
    return config_path, config


def validate_arguments(parser, args):
    if args.rho < 0.0:
        parser.error("the selected rho must be nonnegative")
    if args.translation_rho is not None and args.translation_rho < 0.0:
        parser.error("the selected translation_rho must be nonnegative")
    if not 0.0 <= args.pair_confidence_weight_floor <= 1.0:
        parser.error("the selected pair-confidence floor must be in [0, 1]")
    if args.pair_confidence_weight_power <= 0.0:
        parser.error("the selected pair-confidence power must be positive")


def parse_args(argv=None):
    parser = build_parser()
    runtime = parser.parse_args(argv)
    config_path, selected = load_selected_config(runtime.dataset)

    resolved = dict(selected)
    resolved.update(vars(runtime))
    resolved["best_config_path"] = str(config_path)
    args = argparse.Namespace(**resolved)

    configure_eval_metrics(args)
    validate_arguments(parser, args)
    return args


def main():
    args = parse_args()
    print("Method: GMPCR with L2-normalized SCG Rayleigh confidence")
    print(f"Loaded selected config: {args.best_config_path}")
    print("Resolved arguments:", json.dumps(vars(args), sort_keys=True))
    set_random_seed(args.seed)
    Gmpcr(args).run()


if __name__ == "__main__":
    main()
