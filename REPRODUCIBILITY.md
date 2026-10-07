# Reproducibility record

Companion to **Benchmarking Outage-Resilient Probabilistic Forecasting in Air-Quality
Sensor Networks**. This document records the Houston--Harris County benchmark, the public
release contents, and the limits on reproducing the archived manuscript results.

## Study and protocol chronology

The study uses seven Houston--Harris County U.S. EPA Air Quality System (AQS) stations and
five pollutants. The temporal split is 2021--2023 for training, 2024 for validation,
development selection, and calibration, and 2025 for the chronological holdout, with a
168-hour embargo. The benchmark compares ten learned implementations and two deterministic
baselines using five optimization seeds. Its primary controlled conditions remove the
final 6 or 24 hours of one target station's input history; additional conditions assess
other missingness patterns.

The study is framed as a benchmark, not as a proposal of a new winning model. The
development-selected `ignnk_style_impute_tcn` is a statistical reference for paired
comparisons, not a proposed method and not an exact reproduction of IGNNK. The framing was
amended after development and before the Houston 2025 holdout was downloaded; it is not
prospective preregistration. The paper reports the negative/near-null development findings,
probabilistic trade-offs, undercoverage at 24-hour outages, and the scarcity of natural-gap
cases. The four observed 24-hour natural co-missingness cases do not establish verified
sensor or communications failures.

## Public release contents

The repository includes:

- `src/aqriskformer/`: preprocessing, models, probabilistic scoring, outage evaluation,
  and statistical utilities.
- `scripts/`: Houston data preparation, development, frozen holdout execution/analysis,
  auditing, and the one-command development runner.
- `journal_protocol/`: Houston protocols, amendments, and hash-freeze records.
- `data_external/epa_aqs_houston/`: prepared development and 2025 holdout tensors,
  availability summaries, preprocessing reports, and provenance manifests.
- `tests/`: focused implementation and protocol tests.
- `DATA_SHA256SUMS`: SHA-256 checksums for the distributed prepared tensors.

The public release excludes the archived trained checkpoints, frozen calibration arrays,
saved predictions, per-run metrics, result tables and figures, manuscript, and publication
asset-generation code. Thus the public package supports code inspection and fresh
development experiments, but does not contain all artifacts needed for exact replay of the
manuscript's reported holdout scores.

## Environment and checks

Python 3.10--3.12 is supported. Install the package and development dependencies from the
repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Verify the distributed data and inspect the locked plan before any new fit:

```powershell
Get-FileHash data_external\epa_aqs_houston\prepared_development\*.npz -Algorithm SHA256
Get-FileHash data_external\epa_aqs_houston\prepared_holdout\*.npz -Algorithm SHA256
python scripts\run_journal_development.py --stage plan `
  --protocol journal_protocol\third_comparison_protocol.json
python -m pytest -q
```

Compare the reported hashes with `DATA_SHA256SUMS`. The plan command inspects the
development specification; do not use 2025 labels for new model, threshold, or calibration
selection.

## Fresh development experiments

To fit and analyze fresh development runs in a new output directory:

```powershell
.\scripts\run_houston_development.ps1 -Device cuda:0 `
  -OutputRoot experiment_protocol\reproduction
```

Select another available GPU with `-Device cuda:N`, or use `-Device cpu` where runtime
allows. The runner refuses to overwrite an existing output directory. Fresh fits are not
expected to reproduce archived checkpoint bytes or numerical results exactly because of
hardware, software, and optimization nondeterminism.

The historical freeze-generation scripts record how earlier gates were produced. They
depend on archived intermediate run artifacts that are not part of this public release and
are not part of the fresh-run workflow. The holdout history includes technical amendments
for loader repairs made before inference and an analysis-path repair made after the frozen
evaluations but before metric inspection. These records state the before/after hashes and
whether scientific invariants changed. Consequently, the earliest freeze records contain
superseded source hashes; verify current source against the latest applicable resume or
analysis freeze, and use the amendments to follow the hash chain. Do not regenerate an
archived freeze from a fresh run and present it as the original record.

## Holdout replay boundary

Houston 2025 was evaluated once under the frozen protocol. The public release includes its
prepared tensor for transparency, but omits the matching archived checkpoints, calibration
arrays, predictions, and per-run metric files. Therefore, the holdout runner and analysis
script cannot reproduce the manuscript's exact holdout values from this repository alone;
they require the matching full experiment archive identified by the hash records.

Because the 2025 period and labels are now public, evaluating newly trained models on it is
a rerun of an observed holdout, not an independent confirmation. Do not use those labels to
select models, thresholds, or calibration factors, and do not describe a fresh rerun as a
new external validation.

## Data and software attribution

The measurements originate from the U.S. EPA Air Quality System. EPA states that ambient
monitoring data are public domain and can be reused without a permission request. Cite the
original AQS source and document the filtering, station selection, transformations, and
split boundaries in `DATA.md` and the accompanying paper. The software is distributed under
the MIT license in `LICENSE`; this does not change the public-domain status or attribution
requirements of the source data.
