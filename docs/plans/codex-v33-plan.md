# v33 Enhancement Sequence

Created: 2026-04-06

## Goal

Address the two remaining open Tier 3 code-quality items from the 2026-04-05
peer review: `config.py` modularization (Tier 3.2) and expanded mypy coverage
(Tier 3.4).

## Versioning Approach

The `v32` sequence closed out all Tier 4 diagnostic items.  This sequence
starts at `v33.0` for the code-quality refactor.

- `v33.0` - split `config.py` into a `config/` package (Tier 3.2)
- `v33.1` - expand mypy coverage in CI to 10+ modules (Tier 3.4)
- `v33.2` - refresh peer-review status against landed `v33` work

## v33.0 Scope

`v33.0` splits the 562-line monolithic `config.py` into a `config/` package
with four logical sub-modules and a backward-compatible `__init__.py`.

### Sub-module split

| File | Contents |
|------|----------|
| `config/api.py` | API credentials (AV, FRED, FMP), base URLs, EDGAR helpers, rate limits, HTTP retry, data freshness thresholds |
| `config/features.py` | FRED series lists, publication lags, EDGAR filing lag, ETF benchmark universe + launch dates + proxy map, peer ticker universe, TLH replacement map, feature sets (FEATURES_TO_DROP, MODEL_FEATURE_OVERRIDES), data paths, DB path, ticker, corporate actions |
| `config/model.py` | WFO parameters, Kelly sizing, ensemble models, CPCV, Black-Litterman, fractional diff, diagnostic thresholds (DIAG_*, VIF_*), calibration, conformal prediction, recommendation-layer constants, V13 shadow settings, BLP parameters |
| `config/tax.py` | LTCG/STCG rates, RSU vesting schedule, STCG boundary guard, TLH parameters |
| `config/__init__.py` | Calls `load_dotenv()` first, then re-exports everything via relative `from .submodule import *` |

### Backward compatibility guarantee

All 102 source files use `import config` and access names as `config.XXX`.
The `config/__init__.py` re-exports place every original name in the `config`
namespace, so no call site needs to change.

### Steps

1. Create `config/api.py`, `config/features.py`, `config/model.py`, `config/tax.py`
2. Create `config/__init__.py` with `load_dotenv()` + star imports
3. Delete `config.py`
4. Run full test suite to confirm no regressions

## v33.1 Scope

`v33.1` expands the CI mypy pass from 3 modules to 10+ modules.

Current CI target:
```
src/database/migration_runner.py
src/ingestion/provider_registry.py
src/reporting/run_manifest.py
```

Add:
```
src/research/evaluation.py
src/processing/feature_engineering.py
src/models/drift_monitor.py
src/models/conformal.py
src/reporting/backtest_report.py
src/models/wfo_engine.py
config/model.py
config/tax.py
```

These modules were either recently modified (v31–v33) or are high-value
correctness targets.  Any mypy errors found are fixed in-place as part of
this version.

## v33.2 Scope

`v33.2` keeps the planning layer accurate:

- update 2026-04-05 peer-review status to mark Tier 3.2 and 3.4 as Completed
- refresh remaining-gaps section to point at Tier 4.5 (Monte Carlo tax) and
  Tier 5 strategic items
