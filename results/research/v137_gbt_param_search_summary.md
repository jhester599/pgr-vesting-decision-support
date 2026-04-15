# v137 GBT Parameter Search Summary

Bounded autonomous sweep run on 2026-04-14 against the standalone GBT research frame.

Best observed candidate: `max_depth=1`, `n_estimators=25`, `learning_rate=0.05`, `subsample=0.8` -> R2 `-0.2675`, IC `0.0924`, hit_rate `0.6829`.

Sweep results:
- `depth=1, trees=25, lr=0.05, subsample=0.8` -> R2 `-0.2675`, IC `0.0924`, hit_rate `0.6829`
- `depth=1, trees=50, lr=0.05, subsample=0.8` -> R2 `-0.2966`, IC `0.1310`, hit_rate `0.6724`
- `depth=1, trees=100, lr=0.05, subsample=0.8` -> R2 `-0.3244`, IC `0.1595`, hit_rate `0.6619`
- `depth=1, trees=50, lr=0.05, subsample=0.6` -> R2 `-0.3283`, IC `0.1305`, hit_rate `0.6695`
- `depth=1, trees=50, lr=0.05, subsample=1.0` -> R2 `-0.3307`, IC `0.1005`, hit_rate `0.6762`
- `depth=2, trees=25, lr=0.05, subsample=0.8` -> R2 `-0.3319`, IC `0.0999`, hit_rate `0.6552`
- `depth=1, trees=50, lr=0.1, subsample=0.8` -> R2 `-0.3354`, IC `0.1507`, hit_rate `0.6705`
- `depth=1, trees=100, lr=0.1, subsample=0.8` -> R2 `-0.4139`, IC `0.1410`, hit_rate `0.6504`
- `depth=2, trees=100, lr=0.05, subsample=0.8` -> R2 `-0.4275`, IC `0.1149`, hit_rate `0.6609`
- `depth=2, trees=50, lr=0.1, subsample=0.8` -> R2 `-0.4629`, IC `0.1040`, hit_rate `0.6485`
