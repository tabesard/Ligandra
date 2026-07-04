"""End-to-end orchestrator.

Runs an :class:`~ligandra.config.schema.ExperimentConfig` through
``data -> curate -> split -> train/benchmark -> generate -> score -> rank ->
export -> track``.  The Streamlit UI and the CLI both call these functions, so
"everything the UI does, the API/CLI can do too".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ligandra.config.schema import ExperimentConfig
from ligandra.core.molecule import MoleculeSet
from ligandra.core.seeding import set_global_seed
from ligandra.core.types import ActivityLabel, Column
from ligandra.curate import curate, make_split
from ligandra.curate.curator import CurationReport
from ligandra.data import DATA_SOURCES
from ligandra.featurize import build_featurizer
from ligandra.generate import build_generator, generation_metrics
from ligandra.models import Leaderboard, LeaderboardEntry, build_model
from ligandra.models.base import PredictiveModel
from ligandra.score import build_scorer
from ligandra.score.base import WeightedSumObjective
from ligandra.tracking import build_tracker, hash_dataset


@dataclass
class PipelineResult:
    config: ExperimentConfig
    curated: pd.DataFrame
    curation_report: CurationReport
    leaderboard: Leaderboard
    best_model_name: str | None = None
    best_featurizer_name: str | None = None
    candidates: pd.DataFrame | None = None
    generation_metrics: dict = field(default_factory=dict)
    run_dir: Path | None = None
    dataset_hash: str | None = None


# --------------------------------------------------------------------- data
def _fetch_raw(config: ExperimentConfig) -> pd.DataFrame:
    """Fetch (or read, for local) the raw activities table for the target."""
    t = config.target
    if t.source == "local":
        if not t.path:
            raise ValueError("Local source requires config.target.path.")
        source = DATA_SOURCES.create(
            "local",
            path=t.path,
            smiles_col=t.smiles_col,
            value_col=t.value_col,
            id_col=t.id_col,
            endpoint=t.endpoint.value,
        )
        return source.fetch_activities()

    source = DATA_SOURCES.create(t.source)
    # Treat an empty/blank target_id as "not provided" (the UI stores "" for an
    # empty text box) and fall back to searching by name/query.
    target_id = (t.target_id or "").strip() or None
    if target_id is None:
        query = (t.query or "").strip()
        if not query:
            raise ValueError("Provide config.target.target_id or config.target.query.")
        # select_target picks the best-populated target for the query+endpoint
        # (e.g. human FGFR1, not the FRS2 adaptor that text relevance ranks first).
        target = source.select_target(query, t.endpoint.value)
        if target is None:
            raise ValueError(f"No targets found for query {query!r}.")
        target_id = target.target_id
    return source.fetch_activities(target_id, t.endpoint.value)


def load_and_curate(config: ExperimentConfig) -> tuple[pd.DataFrame, CurationReport]:
    """Fetch (or read from the on-disk cache) activities, then curate.

    Both the raw pull and the curated table are saved as CSVs under
    ``config.cache.dir`` (default ``data_cache/``).  A later run with the same
    target/endpoint reads the raw CSV instead of re-fetching from the network —
    surviving app restarts — and the files are a ready export for analysis.
    """
    from ligandra.data.cache import (
        curated_cache_path,
        load_dataframe,
        raw_cache_path,
        save_dataframe,
        short_hash,
    )

    t = config.target
    cache = config.cache
    target_key = t.target_id or t.query or t.path or "none"
    # a live network fetch is worth caching; a local CSV read already is a file
    use_raw_cache = cache.enabled and t.source != "local"

    raw = None
    raw_path = raw_cache_path(t.source, target_key, t.endpoint.value, cache.dir)
    if use_raw_cache and not cache.refresh:
        raw = load_dataframe(raw_path)
    if raw is None:
        raw = _fetch_raw(config)
        if use_raw_cache and not raw.empty:
            save_dataframe(raw, raw_path)

    if raw.empty:
        raise ValueError(
            f"Data source {t.source!r} returned no activities for target "
            f"{target_key!r} with endpoint {t.endpoint.value!r}. "
            "Check that the target id is correct (for ChEMBL, a UniProt accession "
            "like 'P00533' or a gene/protein name is resolved automatically) and "
            "that this endpoint exists for the target — try another endpoint "
            "(Ki, Kd, EC50) or clear target_id to search by name."
        )

    curated, report = curate(raw, config.target.endpoint, config.curation)

    if cache.enabled and not curated.empty:
        sig = short_hash(config.curation.model_dump(mode="json"))
        save_dataframe(
            curated,
            curated_cache_path(t.source, target_key, t.endpoint.value, sig, cache.dir),
        )
    return curated, report


# ------------------------------------------------------------- model inputs
def _prepare_inputs(model: PredictiveModel, featurizer, mols: MoleculeSet):
    """Give a model the representation it declares it consumes."""
    if getattr(model, "consumes_smiles", False):
        return [m.canonical_smiles or m.smiles for m in mols]
    return featurizer.transform(mols)


def _featurizers_for(model_name: str, config: ExperimentConfig) -> list[str | None]:
    """Choose which featurizer(s) to pair a model with."""
    cls = None
    try:
        from ligandra.models import MODELS

        cls = MODELS.get(model_name)
    except Exception:
        pass
    if cls is not None and getattr(cls, "consumes_smiles", False):
        return [None]  # tokenizer is the featurizer
    if cls is not None and getattr(cls, "consumes_graphs", False):
        return ["graph"]
    return [f.name for f in config.featurizers]


# ------------------------------------------------------------- train / bench
def train_and_benchmark(
    config: ExperimentConfig, curated: pd.DataFrame
) -> tuple[Leaderboard, dict]:
    """Train every (featurizer x model) combo; return leaderboard + fitted best.

    Models are evaluated on the held-out split (scaffold by default).  Returns
    the leaderboard and a dict with the best fitted model/featurizer (refit on
    all data) for reuse as the generative potency objective.
    """
    endpoint = config.target.endpoint
    p_name = endpoint.p_name
    smiles = curated[Column.SMILES].tolist()
    y = curated[p_name].to_numpy(dtype=float)

    split = make_split(
        config.split.strategy.value,
        smiles,
        test_size=config.split.test_size,
        val_size=config.split.val_size,
        seed=config.split.seed,
    )
    all_mols = MoleculeSet.from_smiles(smiles)
    train_mols = MoleculeSet([all_mols[i] for i in split.train])
    test_mols = MoleculeSet([all_mols[i] for i in split.test])
    y_train, y_test = y[split.train], y[split.test]

    board = Leaderboard(task=config.task)
    fitted: dict[str, tuple[PredictiveModel, object, float]] = {}

    for mcfg in config.models:
        for fname in _featurizers_for(mcfg.name, config):
            featurizer = build_featurizer(fname) if fname else None
            fcfg = next((f for f in config.featurizers if f.name == fname), None)
            if fcfg and fname:
                featurizer = build_featurizer(fname, **fcfg.params)
            model = build_model(mcfg.name, **mcfg.params)
            try:
                X_train = _prepare_inputs(model, featurizer, train_mols)
                X_test = _prepare_inputs(model, featurizer, test_mols)
                if mcfg.finetune and mcfg.pretrained:
                    model.finetune(mcfg.pretrained, X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                metrics = model.evaluate(X_test, y_test)
            except (ImportError, NotImplementedError) as exc:
                # Optional heavy deps (torch/transformers) missing: skip cleanly.
                board.add(
                    LeaderboardEntry(mcfg.name, fname or "tokenizer", {"error": str(exc)[:80]})
                )
                continue
            board.add(
                LeaderboardEntry(
                    mcfg.name, fname or "tokenizer", metrics,
                    checkpoint=mcfg.pretrained,
                )
            )
            primary = metrics.get(board.primary_metric, float("nan"))
            fitted[f"{mcfg.name}::{fname}"] = (model, featurizer, primary)

    # Best model refit on ALL data to serve as the potency objective.
    best = _select_best(board, fitted, config)
    return board, best


def _select_best(board: Leaderboard, fitted: dict, config: ExperimentConfig) -> dict:
    df = board.to_dataframe()
    if df.empty or board.primary_metric not in df.columns:
        return {}
    valid = df.dropna(subset=[board.primary_metric])
    if valid.empty:
        return {}
    top = valid.iloc[0]
    key = f"{top['model']}::{top['featurizer'] if top['featurizer'] != 'tokenizer' else None}"
    if key not in fitted:
        return {}
    model, featurizer, _ = fitted[key]
    return {"model": model, "featurizer": featurizer, "model_name": top["model"], "featurizer_name": top["featurizer"]}


# ------------------------------------------------------------- generate/rank
def generate_and_rank(
    config: ExperimentConfig,
    curated: pd.DataFrame,
    best: dict,
) -> tuple[pd.DataFrame, dict]:
    """Generate target-optimized molecules and rank them by the objective."""
    endpoint = config.target.endpoint
    p_name = endpoint.p_name

    # Seed the generator with the most potent training molecules.
    if "label" in curated and (curated["label"] == ActivityLabel.ACTIVE.value).any():
        seed_df = curated[curated["label"] == ActivityLabel.ACTIVE.value]
    else:
        seed_df = curated.nlargest(min(50, len(curated)), p_name)
    seed_smiles = seed_df[Column.SMILES].tolist()
    train_smiles = set(curated[Column.SMILES].tolist())

    generator = build_generator(
        config.generator.name, seed_smiles=seed_smiles, **config.generator.params
    )

    # Build the multi-objective scorer from config.
    terms = []
    scorer_objs: dict[str, object] = {}
    for ocfg in config.objectives:
        scorer = build_scorer(ocfg.scorer, **ocfg.params)
        if ocfg.scorer == "potency":
            if not best:
                continue  # no trained model to score potency
            scorer.bind(best["model"], best["featurizer"])
        if ocfg.scorer == "novelty":
            scorer.set_reference(list(train_smiles))
        terms.append((scorer, ocfg.weight, ocfg.minimize))
        scorer_objs[ocfg.scorer] = scorer
    objective = WeightedSumObjective(terms)

    candidates = generator.optimize_for_target(objective, budget=config.generator.budget)

    # De-duplicate against training actives; keep valid novel molecules.
    kept = []
    seen = set()
    for m in candidates:
        cs = m.canonical_smiles
        if not m.is_valid or cs is None or cs in train_smiles or cs in seen:
            continue
        seen.add(cs)
        kept.append(m)
        if len(kept) >= config.generator.n_output:
            break
    cand_set = MoleculeSet(kept)
    gm = generation_metrics(candidates, reference_smiles=train_smiles).as_dict()

    if len(cand_set) == 0:
        return pd.DataFrame(), gm

    # Per-candidate score breakdown + predicted potency + uncertainty.
    rows = {"smiles": [m.canonical_smiles for m in cand_set]}
    rows["objective"] = objective(cand_set)
    for name, scorer in scorer_objs.items():
        rows[name] = scorer(cand_set)
    if best:
        X = _prepare_inputs(best["model"], best["featurizer"], cand_set)
        mean, std = best["model"].predict_with_uncertainty(X)
        rows[f"pred_{p_name}"] = mean
        rows[f"pred_{p_name}_std"] = std
    df = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    return df, gm


# ------------------------------------------------------------------- driver
def run_experiment(config: ExperimentConfig, do_generate: bool = True) -> PipelineResult:
    """Run the whole pipeline for ``config`` and return a :class:`PipelineResult`."""
    set_global_seed(config.seed)
    tracker = build_tracker(
        config.tracking.backend,
        experiment_name=config.tracking.experiment_name,
        output_dir=config.tracking.output_dir,
    ).start_run(config.name, config.model_dump(mode="json"))

    curated, report = load_and_curate(config)
    ds_hash = hash_dataset(
        curated[Column.SMILES].tolist(),
        curated[config.target.endpoint.p_name].tolist(),
    )
    tracker.log_params({"dataset_hash": ds_hash, "n_curated": len(curated)})
    tracker.log_metrics({f"curation_{k}": v for k, v in report.as_dict().items() if isinstance(v, (int, float))})

    board, best = train_and_benchmark(config, curated)
    lb_df = board.to_dataframe()
    if not lb_df.empty:
        top = lb_df.iloc[0].to_dict()
        tracker.log_metrics({f"best_{k}": v for k, v in top.items() if isinstance(v, (int, float))})

    result = PipelineResult(
        config=config,
        curated=curated,
        curation_report=report,
        leaderboard=board,
        best_model_name=best.get("model_name"),
        best_featurizer_name=best.get("featurizer_name"),
        dataset_hash=ds_hash,
    )

    if do_generate:
        cand_df, gm = generate_and_rank(config, curated, best)
        result.candidates = cand_df
        result.generation_metrics = gm
        tracker.log_metrics({f"gen_{k}": v for k, v in gm.items() if isinstance(v, (int, float))})

    # Persist artifacts.
    run_dir = tracker.run_dir if hasattr(tracker, "run_dir") else None
    if run_dir is not None:
        curated.to_csv(run_dir / "curated.csv", index=False)
        board.save(run_dir / "leaderboard.json")
        tracker.log_artifact(run_dir / "curated.csv")
        tracker.log_artifact(run_dir / "leaderboard.json")
        if result.candidates is not None and not result.candidates.empty:
            result.candidates.to_csv(run_dir / "candidates.csv", index=False)
            tracker.log_artifact(run_dir / "candidates.csv")

    result.run_dir = tracker.end_run()
    return result
