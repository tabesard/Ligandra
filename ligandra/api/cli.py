"""Command-line interface: ``ligandra run experiment.yaml`` (and friends).

The CLI runs the *identical* config the UI produces, headless.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ligandra.api import export_candidates, registries, run
from ligandra.config.schema import ExperimentConfig


def _cmd_run(args: argparse.Namespace) -> int:
    config = ExperimentConfig.from_yaml(args.config)
    if args.no_generate:
        config.generator.budget = 0
    result = run(config, do_generate=not args.no_generate)

    print(f"\n=== Leaderboard ({config.split.strategy.value} split) ===")
    lb = result.leaderboard.to_dataframe()
    print(lb.to_string(index=False) if not lb.empty else "(no models trained)")

    if result.candidates is not None and not result.candidates.empty:
        print(f"\n=== Top candidates (of {len(result.candidates)}) ===")
        print(result.candidates.head(10).to_string(index=False))
        print(f"\nGeneration metrics: {result.generation_metrics}")
        if args.export:
            out = export_candidates(result.candidates, args.export)
            print(f"Exported ranked candidates -> {out}")

    if result.run_dir:
        print(f"\nRun artifacts: {result.run_dir}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    print(json.dumps(registries(), indent=2))
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    cfg = ExperimentConfig(name=args.name)
    path = Path(args.output)
    cfg.to_yaml(path)
    print(f"Wrote starter config -> {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ligandra", description="Target-agnostic CADD platform.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run an experiment from a YAML config.")
    r.add_argument("config", help="Path to experiment YAML.")
    r.add_argument("--export", help="Export ranked candidates to CSV/SDF.")
    r.add_argument("--no-generate", action="store_true", help="Train/benchmark only.")
    r.set_defaults(func=_cmd_run)

    li = sub.add_parser("list", help="List available plugins in every registry.")
    li.set_defaults(func=_cmd_list)

    i = sub.add_parser("init", help="Write a starter experiment config.")
    i.add_argument("-o", "--output", default="experiment.yaml")
    i.add_argument("-n", "--name", default="experiment")
    i.set_defaults(func=_cmd_init)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
