# Test Runtime Optimization Summary

Full-suite baseline measured on 2026-04-14: `131.0s`.

After the first optimization pass:
- full suite: `85.2s` (`35.0%` faster than baseline)
- `--fast` lane: `70.2s` (`46.4%` faster than baseline)

Changes applied in this pass:
- added `scripts/measure_test_time.sh`
- added `--fast` support in `tests/conftest.py`
- registered a `slow` marker in `pytest.ini`
- marked the 5 originally slowest tests with `@pytest.mark.slow`
- cached the pure `v129` evaluator inside the same pytest session
- narrowed several research smoke tests to smaller benchmark subsets where the assertion contract stayed intact

Post-pass slowest tests from `--durations=10`:
- `tests/test_research_v128_benchmark_feature_search.py::test_run_feature_search_smoke_covers_all_benchmarks_with_small_subset` -> ~5.75s
- `tests/test_multi_benchmark_wfo.py::TestRunAllBenchmarks::test_ridge_model_type_stored` -> ~3.39s
- `tests/test_research_v134_fred_lag_sweep.py::test_default_no_override_recovers_v38_style_baseline` -> ~3.13s
- `tests/test_multi_benchmark_wfo.py::TestLoggingFallbacks::test_run_ensemble_benchmarks_logs_failed_model_and_continues` -> ~2.77s
- `tests/test_research_v133_ridge_alpha_sweep.py::test_default_grid_produces_reasonable_r2` -> ~2.71s

Interpretation:
- The plan's `>= 20%` runtime reduction goal is met on the full suite in this first pass.
- The remaining bottlenecks are now concentrated in multi-benchmark WFO integration tests and one benchmark-search smoke test.
