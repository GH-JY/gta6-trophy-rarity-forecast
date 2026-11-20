#!/usr/bin/env python3
"""Forecast Grand Theft Auto VI (PS5) trophy earn rates for make.fun's gta6-trophy-rarity-forecast-01.

The program reads every *.in case in MAKEFUN_INPUT_DIR (default /input) and writes <id>.out to
MAKEFUN_OUTPUT_DIR (default /output).  One case file holds the house's reads of Exophase's trophy page before a
checkpoint; the output is, for every trophy of the case, five quantiles of its earn rate (percent of tracked
players) at the case's target, at the levels 0.05, 0.25, 0.5, 0.75 and 0.95.

The model, step by step:

  * Page states.  Exophase refreshes its counts on its own schedule, so most reads repeat the previous page.
    Consecutive reads whose players_tracked and rate vectors are identical collapse into one state, timed at
    the first slot of the run.  A case of seven identical launch-night reads is one state, not seven.
  * Trend.  Each trophy's logit rate is fitted against x = ln(1 + hours from release midnight UTC) by weighted
    least squares: weight 0.5 ** (age_hours / 36), floored so the earliest states still count a little.  A
    trophy with fewer than three usable states has no slope of its own and borrows the pooled one.
  * Shrinking.  The pooled slope is the weighted mean of the per-trophy fits, guarded to a sane range.  Each
    trophy's slope moves toward it with a weight of four pseudo-states, then scales by 0.85 so extrapolation
    cannot run away.  The line is anchored at the trophy's last observed logit, and the predicted drift from
    it is capped at +-2.5 logit units, which also keeps rates pinned at 100% near the top.
  * Spread.  The scale of a forecast error is estimated from the case itself: every earlier state predicts
    every later one with the same anchored-slope rule on the data available at the earlier state, and the
    residuals are fitted as sigma * sqrt(gap in days).  Fewer than twenty usable pairs falls back to 0.45.
    The target's spread is sigma * sqrt(horizon days), floored at 0.5 and capped at 2.0, plus a small term
    for how far the target lies beyond the last state's x (slope uncertainty).
  * Quantiles.  Inverse logit of the predicted logit plus z * spread, z = -1.645, -0.674, 0, 0.674, 1.645,
    clamped to 0.01..99.99, rounded to four decimals, and made non-decreasing.

Every case is processed inside a try/except: unexpected data falls back to each trophy's last observed rate
with a modest random-walk spread, and the output is always a valid file.  Standard library only, no network,
no randomness, so two runs write identical bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)

HALF_LIFE_HOURS = 36.0      # weight half-life for the trend fit
WEIGHT_FLOOR = 0.05         # oldest states never fall below this weight
PRIOR_STATES = 4.0          # pseudo-states pulling a slope toward the pooled one
SLOPE_SCALE = 0.85          # damping on every slope before it is used
POOLED_SLOPE_CAP = 0.8      # guard on the pooled slope, logit per ln(1+hour)
DRIFT_CAP = 2.5             # largest allowed move from the last observed logit

FALLBACK_SIGMA = 0.45       # logit per sqrt(day) when the backtest is too thin
MIN_BACKTEST_PAIRS = 20
SIGMA_MIN = 0.10
SIGMA_MAX = 1.20
SPREAD_FLOOR = 0.5
SPREAD_CAP = 2.0
EXTRAPOLATION_K = 0.12      # extra spread per log-hour beyond the last state's x
CEILING_LOGIT = math.log(9999.0)  # logit of 99.99, the top of the clamped range
CEILING_SPREAD_CAP = 0.6    # a rate pinned at the top has a censored, one-sided error

MAX_BACKTEST_STATES = 48    # bound the O(states^2 * trophies) backtest
MAX_MODEL_STATES = 240      # bound the trend fit on absurdly long histories
DEFAULT_RATE = 1.0          # percent, only for a trophy never observed at all


def clamp_pct(value: float) -> float:
    """Clamp a percentage into the range logit() is defined on."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0.01
    if value < 0.01:
        return 0.01
    if value > 99.99:
        return 99.99
    return float(value)


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def pct_from_logit(z: float) -> float:
    """Inverse logit, overflow-safe."""
    if z >= 0.0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """Round half up to four decimals; a whole value comes back as an int, so 100 prints as 100."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_epoch(text: str) -> float:
    """ISO 8601 UTC timestamp -> epoch seconds."""
    s = str(text).strip()
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    stamp = datetime.fromisoformat(s)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def release_epoch(case: dict, times: list[float]) -> float:
    """Midnight UTC of the game's release date; the origin of the x axis."""
    game = case.get("game")
    text = game.get("release_date") if isinstance(game, dict) else None
    if isinstance(text, str):
        try:
            day = datetime.fromisoformat(text.strip()[:10])
            return datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
        except Exception:
            pass
    return times[0] if times else 0.0


def x_at(epoch: float, origin: float) -> float:
    """ln(1 + hours since release midnight)."""
    hours = (epoch - origin) / 3600.0
    base = 1.0 + hours
    if base < 1e-3:
        base = 1e-3
    return math.log(base)


def collapse_states(history: list, trophy_count: int) -> tuple[list[list[float | None]], list[float]]:
    """Consecutive identical pages become one state, timed at the first slot of the run."""
    states: list[list[float | None]] = []
    times: list[float] = []
    previous: tuple | None = None
    for read in history:
        if not isinstance(read, dict):
            continue
        raw = read.get("rates")
        if not isinstance(raw, list):
            raw = []
        rates: list[float | None] = []
        for i in range(trophy_count):
            value = raw[i] if i < len(raw) else None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                rates.append(None)
            else:
                number = float(value)
                rates.append(number if math.isfinite(number) else None)
        key = (read.get("players_tracked"), tuple(rates))
        if states and key == previous:
            continue
        stamp = read.get("slot")
        if stamp is None:
            stamp = read.get("at")
        try:
            when = parse_epoch(stamp)
        except Exception:
            if not times:
                continue
            when = times[-1]
        states.append(rates)
        times.append(when)
        previous = key
    return states, times


def sample_indices(n: int, limit: int) -> list[int]:
    """Evenly spaced indices from 0 to n-1, at most `limit` of them."""
    if n <= limit:
        return list(range(n))
    out: list[int] = []
    for i in range(limit):
        index = int(round(i * (n - 1) / (limit - 1)))
        if not out or index != out[-1]:
            out.append(index)
    return out


def thin(times: list[float], states: list, limit: int) -> tuple[list[float], list]:
    """Deterministically thin an over-long state list, always keeping the first and the last."""
    if len(times) <= limit:
        return times, states
    keep = sample_indices(len(times), limit)
    return [times[i] for i in keep], [states[i] for i in keep]


def build_series(
    states: list, times: list[float], origin: float, trophy_count: int
) -> tuple[list[float], list[list[tuple[int, float, float]]]]:
    """Per-trophy (state index, x, logit rate) observations."""
    xs = [x_at(t, origin) for t in times]
    series: list[list[tuple[int, float, float]]] = [[] for _ in range(trophy_count)]
    for k in range(len(states)):
        x = xs[k]
        rates = states[k]
        for i in range(trophy_count):
            rate = rates[i]
            if rate is None:
                continue
            series[i].append((k, x, logit_pct(rate)))
    return xs, series


def weighted_fit(points: list[tuple[float, float, float]]) -> float | None:
    """Weighted least-squares slope of y on x, or None when x has no spread."""
    total = 0.0
    for _, _, w in points:
        total += w
    if total <= 0.0:
        return None
    mean_x = 0.0
    mean_y = 0.0
    for x, y, w in points:
        mean_x += w * x
        mean_y += w * y
    mean_x /= total
    mean_y /= total
    sxx = 0.0
    sxy = 0.0
    for x, y, w in points:
        dx = x - mean_x
        sxx += w * dx * dx
        sxy += w * dx * (y - mean_y)
    if sxx <= 1e-12:
        return None
    return sxy / sxx


def slopes_at(series: list[list[tuple[int, float, float]]], times: list[float], anchor: int) -> list[float]:
    """Shrunk, scaled slopes per trophy using only the states up to `anchor`."""
    t_anchor = times[anchor]
    count = len(series)
    raw: list[float | None] = [None] * count
    total_w = [0.0] * count
    n_obs = [0] * count
    for i in range(count):
        points: list[tuple[float, float, float]] = []
        for k, x, y in series[i]:
            if k > anchor:
                break
            age = (t_anchor - times[k]) / 3600.0
            if age <= 0.0:
                weight = 1.0
            else:
                weight = 0.5 ** (age / HALF_LIFE_HOURS)
                if weight < WEIGHT_FLOOR:
                    weight = WEIGHT_FLOOR
            points.append((x, y, weight))
        n_obs[i] = len(points)
        if len(points) >= 3:
            slope = weighted_fit(points)
            if slope is not None:
                raw[i] = slope
                weight_sum = 0.0
                for point in points:
                    weight_sum += point[2]
                total_w[i] = weight_sum

    numerator = 0.0
    denominator = 0.0
    for i in range(count):
        slope = raw[i]
        if slope is None:
            continue
        numerator += slope * total_w[i]
        denominator += total_w[i]
    pooled = numerator / denominator if denominator > 0.0 else 0.0
    if pooled > POOLED_SLOPE_CAP:
        pooled = POOLED_SLOPE_CAP
    elif pooled < -POOLED_SLOPE_CAP:
        pooled = -POOLED_SLOPE_CAP

    out = [0.0] * count
    for i in range(count):
        slope = raw[i]
        if slope is None:
            value = pooled
        else:
            value = (n_obs[i] * slope + PRIOR_STATES * pooled) / (n_obs[i] + PRIOR_STATES)
        out[i] = value * SLOPE_SCALE
    return out


def estimate_sigma(
    times: list[float], xs: list[float], series: list[list[tuple[int, float, float]]]
) -> float:
    """Backtest every earlier state against every later one; rms residual = sigma * sqrt(gap days)."""
    n = len(times)
    if n < 2:
        return FALLBACK_SIGMA
    sampled = sample_indices(n, MAX_BACKTEST_STATES)
    anchors = [i for i in sampled if i < n - 1]
    targets = [i for i in sampled if i > 0]
    by_index: list[dict[int, float]] = []
    for points in series:
        lookup: dict[int, float] = {}
        for k, _, y in points:
            lookup[k] = y
        by_index.append(lookup)

    residual_sq = 0.0
    gap_sum = 0.0
    pairs = 0
    for a in anchors:
        slopes = slopes_at(series, times, a)
        anchor_values: list[tuple[float, float] | None] = []
        for points in series:
            value: tuple[float, float] | None = None
            for k, x, y in points:
                if k <= a:
                    value = (x, y)
                else:
                    break
            anchor_values.append(value)
        for j in targets:
            if j <= a:
                continue
            gap = (times[j] - times[a]) / 86400.0
            if gap <= 0.0:
                continue
            target_x = xs[j]
            for i in range(len(series)):
                start = anchor_values[i]
                if start is None:
                    continue
                actual = by_index[i].get(j)
                if actual is None:
                    continue
                drift = slopes[i] * (target_x - start[0])
                if drift > DRIFT_CAP:
                    drift = DRIFT_CAP
                elif drift < -DRIFT_CAP:
                    drift = -DRIFT_CAP
                residual = actual - (start[1] + drift)
                residual_sq += residual * residual
                gap_sum += gap
                pairs += 1

    if pairs < MIN_BACKTEST_PAIRS or gap_sum <= 0.0:
        return FALLBACK_SIGMA
    sigma = math.sqrt(residual_sq / gap_sum)
    if not math.isfinite(sigma):
        return FALLBACK_SIGMA
    if sigma < SIGMA_MIN:
        return SIGMA_MIN
    if sigma > SIGMA_MAX:
        return SIGMA_MAX
    return sigma


def make_row(trophy_id: int, predicted_logit: float, spread: float) -> dict:
    """Five quantiles around the predicted logit, clamped, rounded and made non-decreasing."""
    needed = len(LEVELS)
    values: list[float | int] = [0.0] * needed
    previous: float | int = 0.01
    for index in range(needed):
        value = pct_from_logit(predicted_logit + Z[index] * spread)
        value = round4(clamp_pct(value))
        if value < previous:
            value = previous
        if value > 99.99:
            value = 99.99
        previous = value
        values[index] = value
    return {"id": trophy_id, "q": values}


def horizon_days(case: dict, last_time: float) -> float:
    try:
        minutes = float(case.get("horizon_minutes"))
        if math.isfinite(minutes) and minutes > 0.0:
            return minutes / 1440.0
    except (TypeError, ValueError):
        pass
    try:
        span = (parse_epoch(case.get("target")) - last_time) / 86400.0
        if math.isfinite(span) and span > 0.0:
            return span
    except Exception:
        pass
    return 0.5


def target_x(case: dict, xs: list[float], origin: float) -> float:
    try:
        x = x_at(parse_epoch(case.get("target")), origin)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return xs[-1] + 0.5


def model_forecast(case: dict) -> dict:
    """The trend model: states, shrunk slopes, backtested spread, quantiles."""
    trophies = case.get("trophies")
    if not isinstance(trophies, list):
        trophies = []
    count = len(trophies)
    if count == 0:
        return {"trophies": []}

    history = case.get("history")
    if not isinstance(history, list):
        history = []
    states, times = collapse_states(history, count)
    if not states:
        raise ValueError("no usable reads")
    times, states = thin(times, states, MAX_MODEL_STATES)

    origin = release_epoch(case, times)
    xs, series = build_series(states, times, origin, count)
    days = horizon_days(case, times[-1])
    x_target = target_x(case, xs, origin)

    sigma = estimate_sigma(times, xs, series)
    spread = sigma * math.sqrt(days)
    if spread < SPREAD_FLOOR:
        spread = SPREAD_FLOOR
    elif spread > SPREAD_CAP:
        spread = SPREAD_CAP
    if x_target > xs[-1]:
        spread += EXTRAPOLATION_K * (x_target - xs[-1])

    slopes = slopes_at(series, times, len(times) - 1)
    rows: list[dict] = []
    for i in range(count):
        trophy = trophies[i]
        trophy_id = trophy.get("id") if isinstance(trophy, dict) else None
        if trophy_id is None:
            trophy_id = i + 1
        points = series[i]
        if not points:
            rows.append(make_row(trophy_id, logit_pct(DEFAULT_RATE), spread))
            continue
        _, x_last, y_last = points[-1]
        drift = slopes[i] * (x_target - x_last)
        if drift > DRIFT_CAP:
            drift = DRIFT_CAP
        elif drift < -DRIFT_CAP:
            drift = -DRIFT_CAP
        own_spread = spread
        if y_last >= CEILING_LOGIT - 1e-9:
            drift = 0.0
            if own_spread > CEILING_SPREAD_CAP:
                own_spread = CEILING_SPREAD_CAP
        rows.append(make_row(trophy_id, y_last + drift, own_spread))
    return {"trophies": rows}


def fallback_forecast(case: dict) -> dict:
    """Persistence at each trophy's last observed rate, with a modest random-walk spread."""
    trophies = case.get("trophies")
    if not isinstance(trophies, list):
        trophies = []
    count = len(trophies)
    if count == 0:
        return {"trophies": []}

    last_rates: list[float | None] = [None] * count
    history = case.get("history")
    if isinstance(history, list):
        for read in history:
            if not isinstance(read, dict):
                continue
            raw = read.get("rates")
            if not isinstance(raw, list):
                continue
            for i in range(min(count, len(raw))):
                value = raw[i]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                number = float(value)
                if math.isfinite(number):
                    last_rates[i] = number

    days = horizon_days(case, 0.0)
    spread = 0.5 * math.sqrt(days)
    if spread < SPREAD_FLOOR:
        spread = SPREAD_FLOOR
    elif spread > SPREAD_CAP:
        spread = SPREAD_CAP

    rows: list[dict] = []
    for i in range(count):
        trophy = trophies[i]
        trophy_id = trophy.get("id") if isinstance(trophy, dict) else None
        if trophy_id is None:
            trophy_id = i + 1
        rate = last_rates[i]
        rows.append(make_row(trophy_id, logit_pct(DEFAULT_RATE if rate is None else rate), spread))
    return {"trophies": rows}


def forecast_case(case: dict) -> dict:
    """Never raise: the trend model, else persistence, else an empty but valid document."""
    if not isinstance(case, dict):
        case = {}
    try:
        return model_forecast(case)
    except Exception:
        pass
    try:
        return fallback_forecast(case)
    except Exception:
        pass
    return {"trophies": []}


def main() -> None:
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    try:
        paths = sorted(path for path in input_dir.glob("*.in") if path.is_file())
    except Exception:
        paths = []
    for path in paths:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            case = {}
        forecast = forecast_case(case)
        try:
            text = json.dumps(forecast, separators=(",", ":"), ensure_ascii=True) + "\n"
        except Exception:
            text = '{"trophies":[]}\n'
        try:
            (output_dir / (path.stem + ".out")).write_text(text, encoding="utf-8", newline="\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
