# Houston benchmark protocol records

These files document the completed Houston--Harris County benchmark and the code/data
freezes used for the IEEE Sensors Journal manuscript. They contain protocol and hash
metadata; they do not include checkpoints, predictions, metric outputs, tables, or figures.

## Main specifications

- `third_development_preprocessing_protocol.json`: source measurements, network, time axis,
  masks, causal preprocessing, and training-only transforms.
- `third_comparison_protocol.json`: model panel, seeds, optimization, corruptions, endpoints,
  and development-only selection rules.
- `third_benchmark_amendment.json`: transparent post-development, pre-Houston-holdout
  change to benchmark framing. It explicitly does not claim prospective preregistration.
- `third_benchmark_final_holdout_protocol.json`: frozen 2025 comparison and use restrictions.
- `third_benchmark_final_holdout_execution_freeze.json` and
  `third_benchmark_analysis_execution_freeze.json`: source, input, and output hashes for
  the archived execution and analysis.
- `third_benchmark_technical_amendment_*.json` and
  `third_benchmark_technical_resume_*.json`: dated loader/analysis wiring corrections made
  before the corresponding model metrics were computed or opened, as detailed in the
  supplement.

Hash freezes refer to archived local runs. The public release omits those checkpoints,
calibration arrays, predictions, and metrics; see the repository-level
`REPRODUCIBILITY.md` for what can and cannot be replayed from the public files alone.
