# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/GH-JY/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/GH-JY/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic, standard-library-only Python 3.13 forecaster for the make.fun GTA VI trophy-rarity bounty. For each case it collapses repeated page reads into page states, fits every trophy's earn rate in logit space against ln(1 + hours since release midnight UTC) with a 36-hour recency half-life, shrinks sparse per-trophy slopes toward a pooled slope, anchors the trend at each trophy's last observed rate, and backtests earlier states against later ones to estimate how quickly an anchored prediction loses accuracy (sigma per square-root day, with a 0.45 fallback when the history is too thin). The output is five quantiles per trophy at the 0.05, 0.25, 0.5, 0.75 and 0.95 levels, clamped to 0.01..99.99 percent; a rate pinned at 100 percent stays near the top, and any unexpected input falls back to each trophy's last observed rate with a modest spread so a valid file is always written.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
