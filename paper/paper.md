---
title: 'Ligandra: a target-agnostic, extensible platform for computer-aided drug design'
tags:
  - Python
  - cheminformatics
  - drug design
  - QSAR
  - de novo molecular generation
  - machine learning
  - transfer learning
authors:
  - name: FIRST LAST            # <-- replace with your name
    orcid: 0000-0000-0000-0000  # <-- replace with your ORCID
    affiliation: 1
affiliations:
  - name: YOUR INSTITUTION, City, Country   # <-- replace
    index: 1
date: 3 July 2026
bibliography: paper.bib
---

# Summary

`Ligandra` is an open-source Python platform for computer-aided drug design
(CADD) that runs an entire early-discovery workflow — data collection, curation,
featurization, predictive (QSAR/QSPR) modelling, transfer learning from
pretrained molecular foundation models, de-novo molecular generation, and
multi-objective scoring and ranking — from a single typed configuration. Its
defining design choice is that every stage (data sources, featurizers, predictive
models, generators, scoring functions, docking engines, and pretrained
checkpoints) is a plugin discovered through a string-keyed registry behind a small
abstract interface. Adding a new model or generator is therefore one subclass and
one decorator: it appears in the command-line tool, the Python API, and the
Streamlit user interface with no changes to orchestration code. The same
configuration object drives all three entry points, so a workflow explored
interactively in the UI can be re-run headless and reproducibly.

Ligandra ships classical models (linear/ridge/lasso, random forest, SVR, and
optional gradient boosting), a graph-neural-network wrapper, transformer
foundation-model fine-tuning, and a spectrum of generative engines (a graph
genetic algorithm requiring no deep-learning dependencies, a SMILES language
model with reinforcement learning, a variational autoencoder with latent-space
optimization, a Transformer decoder, and an E(3)-equivariant 3D diffusion model),
built on RDKit [@rdkit] and scikit-learn [@scikit-learn] with optional PyTorch
[@pytorch] and Hugging Face Transformers [@transformers] back ends. Molecules are
scored on predicted potency, drug-likeness (QED), synthetic accessibility,
novelty and diversity, combined into a configurable weighted desirability and a
Pareto front.

# Statement of need

Practitioners assembling a CADD pipeline today typically stitch together several
excellent but separate libraries — RDKit for chemistry [@rdkit], ChEMBL for
bioactivity data [@chembl], scikit-learn or PyTorch for models, and dedicated
tools such as REINVENT for generation [@reinvent] — writing bespoke glue for each
new target, endpoint, descriptor set, or model. Existing integrated toolkits
(for example DeepChem [@deepchem] and TorchDrug [@torchdrug]) provide rich model
zoos but leave the end-to-end, target-to-ranked-candidates workflow and its
reproducibility to the user. Ligandra's contribution is not a new algorithm but a
**coherent, config-driven architecture** that makes this workflow target-agnostic
and one-class extensible, and that surfaces the same pipeline through a guided UI,
a CLI, and an API.

A second, more technical contribution addresses a subtle correctness hazard in
transfer learning. When fine-tuning a pretrained molecular foundation model such
as ChemBERTa [@chemberta] or MoLFormer [@molformer], downstream molecules must be
encoded through the *exact* preprocessing the checkpoint saw during pretraining
(the same standardization, tokenizer, vocabulary and padding); otherwise
embeddings are out-of-distribution and fine-tuning silently learns nothing
transferable, even as training loss decreases. Ligandra makes this a bound
contract: each checkpoint carries a `PreprocessingSpec` and a stable
`preprocess_hash`, downstream data is encoded only through that spec, and a guard
refuses to mix pipelines across the fine-tune set, later inference, and any
generator that reuses the model. The platform also fetches only the small
target-specific dataset for fine-tuning (never re-assembling a pretraining
corpus), deduplicates against known sources by InChIKey, and reports an
applicability-domain flag — behaviours that are enforced by the test suite.

Ligandra targets computational chemists and machine-learning researchers who need
a reproducible scaffold for target-specific model building and de-novo design,
and educators who need a transparent, end-to-end teaching platform. It is
explicitly a research / early-discovery tool: predictions and generated molecules
are hypotheses for experimental validation, honest scaffold-split evaluation is
the default, and reported uncertainty and novelty are foregrounded throughout.

# Design and features

- **Config-driven core.** A single Pydantic model fully describes an experiment
  (target, endpoint, curation thresholds, featurizers, models, generator,
  objectives, budgets, caching and tracking) and serializes to YAML.
- **Registry + plugin pattern** for all seven layers, with matching abstract base
  classes; new plugins require no orchestration edits.
- **Target-agnostic data ingestion** from ChEMBL (resolving ChEMBL ids, UniProt
  accessions, or gene/protein names), with local CSV/SDF upload, on-disk caching,
  and automatic retry on transient network errors.
- **Honest evaluation** via Bemis–Murcko scaffold splitting and a model
  leaderboard, with applicability-domain and uncertainty estimates.
- **Correct transfer learning** through the representation-consistency contract
  described above.
- **Reproducibility**: global seeding, dataset hashing, per-run manifests, and
  optional MLflow tracking.
- **Three coequal front ends** — Streamlit UI, CLI (`ligandra run experiment.yaml`),
  and Python API — over one library, with generated candidates rendered as 2D
  structures annotated with predicted potency.

# Acknowledgements

We thank the maintainers of RDKit, scikit-learn, ChEMBL, PyTorch and Hugging Face
Transformers, on which Ligandra builds.

# References
