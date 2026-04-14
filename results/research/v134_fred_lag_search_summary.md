# v134 FRED Lag Search Summary

Bounded autonomous sweep run on 2026-04-14 against the current 8-benchmark production ensemble research frame.

Baseline all-ones candidate: `R2=-0.1578`, `IC=0.1261`, `hit_rate=0.6906`.
All-zero candidate: `R2=-0.1657`, `IC=0.1149`, `hit_rate=0.7011`.

Best observed candidate:
- overrides: `{"BAA10Y": 1, "BAMLH0A0HYM2": 1, "GS10": 1, "GS2": 1, "GS5": 1, "MORTGAGE30US": 1, "T10Y2Y": 1, "T10YIE": 0, "VIXCLS": 1}`
- pooled_oos_r2: `-0.1573`
- pooled_ic: `0.1262`
- pooled_hit_rate: `0.6983`

Single-toggle results:
- `T10YIE` -> R2 `-0.1573`, IC `0.1262`, hit_rate `0.6983`
- `BAA10Y` -> R2 `-0.1578`, IC `0.1261`, hit_rate `0.6906`
- `MORTGAGE30US` -> R2 `-0.1578`, IC `0.1261`, hit_rate `0.6906`
- `VIXCLS` -> R2 `-0.1585`, IC `0.1206`, hit_rate `0.6925`
- `T10Y2Y` -> R2 `-0.1613`, IC `0.1349`, hit_rate `0.6858`
- `GS5` -> R2 `-0.1627`, IC `0.1216`, hit_rate `0.6887`
- `BAMLH0A0HYM2` -> R2 `-0.1676`, IC `0.1290`, hit_rate `0.6973`
- `GS2` -> R2 `-0.1686`, IC `0.1141`, hit_rate `0.6954`
- `GS10` -> R2 `-0.1834`, IC `0.0951`, hit_rate `0.6772`
