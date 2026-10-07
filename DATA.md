# Houston AQS data

## Source and scope

The Houston--Harris County observations originate from the U.S. Environmental Protection
Agency (EPA) Air Quality System (AQS), state FIPS `48`, county FIPS `201`. The network uses
seven fixed stations:

`48-201-1039`, `48-201-1034`, `48-201-0024`, `48-201-0066`, `48-201-0046`,
`48-201-1052`, and `48-201-0058`.

The five AQS parameter codes are PM$_{2.5}$ (`88101`), NO$_2$ (`42602`), O$_3$ (`44201`),
CO (`42101`), and SO$_2$ (`42401`). Pollutant support differs by station; the stored
observation and target masks must be respected.

EPA identifies ambient AQS monitoring measurements as public domain. The data may be
downloaded and reused without requesting permission; cite the EPA AQS source in derivative
work. See the [EPA reuse statement](https://www.epa.gov/outdoor-air-quality-data/do-i-need-request-permission-use-monitoring-data-and-graphics-airdata).
The software code in this repository is separately distributed under the MIT license in
`LICENSE`.

## Included files

```text
data_external/epa_aqs_houston/prepared_development/
  houston_2021_2024.npz
  availability.csv
  preprocessing_report.json
data_external/epa_aqs_houston/prepared_holdout/
  houston_2025.npz
  availability.csv
  preparation_report.json
data_external/epa_aqs_houston/provenance/
  development_content_audit.json
  development_download_manifest.json
  holdout_download_manifest.json
```

The prepared tensors are deterministic inputs for the published protocol. Original
compressed API responses are not duplicated. Download manifests record their provenance
and hashes; they contain no API key. A raw-data download requires an EPA AQS API email and
key supplied by the user. Never commit personal credentials.

## Chronology

- Development covers 2021--2024. Training uses 2021--2023; 2024 supplies validation,
  development selection, and calibration.
- Houston 2025 is a chronological holdout. Its first scored target follows a 168-hour
  embargo at the start of the year.
- The 2025 prepared tensor is included because the final benchmark has already been
  completed and the source observations are public. Do not use it for new model selection,
  threshold selection, or calibration.

The experiment uses a continuous hourly UTC axis. Calendar features use fixed local
standard time (UTC-06:00), without daylight-saving shifts. Only one-hour AQS sample-duration
records are retained. Concurrent finite observations at the same station, pollutant, and
hour are combined by their median. Missing values remain absent in the native mask; input
fills are causal. Scaling, event thresholds, and MASE scales are estimated from training
data only.

## NPZ schema

Load prepared development data with `aqriskformer.data.PreparedAirQuality.load`. The
archives retain timestamps, station and pollutant identifiers, model inputs, native
observation and target masks, unscaled targets, calendar and station features, training-only
scaling parameters, risk thresholds, MASE denominators, and split metadata. Use the masks
when scoring; a missing target is not a zero-valued observation.

## Rebuilding from AQS

The repository includes `scripts/download_epa_aqs.py`,
`scripts/validate_epa_aqs_download.py`, `scripts/audit_epa_aqs_development.py`, and
`scripts/prepare_epa_aqs_development.py`. Their arguments and the fixed station/parameter
scope are recorded in `journal_protocol/third_development_preprocessing_protocol.json`
and `journal_protocol/third_unseen_holdout_protocol.json`. API credentials are supplied
through the user's environment and are not stored in the repository.

`DATA_SHA256SUMS` contains checksums for the two distributed NPZ tensors.
