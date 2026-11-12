#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 (make.fun program/v1).

Reads /input/history.ndjson (hourly-ish snapshots of the chart) and
/input/task.json, and writes one forecast to /output/forecast.json: a
place-by-game probability matrix plus a top-10 guess, scored against
every hourly read of the 72 hours after the deadline.

Model: recency-weighted, rank-kernel-smoothed occupancy. Every past
observation of a game at a rank contributes probability mass to nearby
places (a Gaussian kernel over ranks, because ranks diffuse over time),
with older reads down-weighted exponentially. A per-game trend
multiplier rewards games whose rank and player count have recently
improved; games approaching the chart's 28-day age limit are faded
out using their creation date when the history carries it. Mass is
shrunk slightly toward uniform for calibration, and each place names
only the games whose probability beats the pooled unnamed share, so
the long tail and brand-new debutants stay covered by the unnamed
share instead of being given false precision.

Standard library only, no network, fully deterministic: the same
history always produces the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone

# --- Tuned constants (chosen by backtesting on the kit's history) ---
HALF_LIFE_HOURS = 9.0    # recency decay of past reads
KERNEL_SIGMA = 8.0       # rank-diffusion width, in places
TREND_GAMMA = 0.60       # strength of the momentum multiplier
SHRINK_LAMBDA = 0.40     # shrinkage toward uniform (calibration)
AGE_SOFT_DAYS = 26.5     # age at window midpoint where fading starts
AGE_HARD_DAYS = 28.5     # age at window midpoint where a game is gone
MAX_ROWS_PER_PLACE = 100  # the format's cap
AGE_LIMIT_DAYS = 28.0    # the chart only lists games published <28 days ago


def parse_time(value):
    """Parse an ISO 8601 UTC timestamp to epoch seconds; None on failure."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def weighted_slope(points):
    """Weighted least-squares slope of y vs t (hours) for [(t, y, w)]."""
    sw = sum(w for _, _, w in points)
    if sw <= 0 or len(points) < 3:
        return 0.0
    mt = sum(w * t for t, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    num = sum(w * (t - mt) * (y - my) for t, y, w in points)
    den = sum(w * (t - mt) ** 2 for t, _, w in points)
    if den <= 0:
        return 0.0
    return num / den


def collect_created(details, created):
    """Harvest universe_id -> created-epoch from a snapshot's details blob.

    The recorder's details map each up-and-coming game to its games/v1
    record; tolerate dict- or list-shaped blobs and missing fields.
    """
    if isinstance(details, dict):
        items = details.items()
    elif isinstance(details, list):
        items = [(d.get("universe_id", d.get("id")), d) for d in details
                 if isinstance(d, dict)]
    else:
        return
    for key, record in items:
        if not isinstance(record, dict):
            continue
        try:
            uid = int(key)
        except (TypeError, ValueError):
            continue
        made = parse_time(record.get("created"))
        if made is not None and uid not in created:
            created[uid] = made


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

    with open(os.path.join(input_dir, "task.json"), encoding="utf-8") as fh:
        task = json.load(fh)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")

    snapshots = []
    with open(os.path.join(input_dir, "history.ndjson"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                snapshots.append(json.loads(line))
    snapshots.sort(key=lambda s: parse_time(s.get("at")) or 0.0)

    window = task.get("window") or {}
    t_ref = parse_time(window.get("from"))
    if t_ref is None:
        t_ref = parse_time(snapshots[-1].get("at")) if snapshots else 0.0
    window_until = parse_time(window.get("until"))
    if window_until is None or window_until <= t_ref:
        window_until = t_ref + 72.0 * 3600.0
    t_mid = t_ref + 0.5 * (window_until - t_ref)

    # Per-game observations: (time_hours, rank, playing, weight).
    obs = {}
    created = {}
    chart_size = 0
    for snap in snapshots:
        at = parse_time(snap.get("at"))
        if at is None:
            continue
        age_hours = max(0.0, (t_ref - at) / 3600.0)
        weight = 0.5 ** (age_hours / HALF_LIFE_HOURS)
        rows = snap.get("up_and_coming") or []
        chart_size = max(chart_size, len(rows))
        collect_created(snap.get("details"), created)
        for row in rows:
            try:
                uid = int(row["universe_id"])
                rank = int(row["rank"])
            except (KeyError, TypeError, ValueError):
                continue
            playing = row.get("playing")
            obs.setdefault(uid, []).append((at / 3600.0, rank, playing, weight))

    # Score mass per (place, game): kernel-smoothed, recency-weighted.
    inv_two_sigma2 = 1.0 / (2.0 * KERNEL_SIGMA * KERNEL_SIGMA)
    scores = {k: {} for k in range(1, places + 1)}
    for uid, points in obs.items():
        # Trend: rank slope (negative = climbing) and player-growth slope.
        rank_slope = weighted_slope([(t, r, w) for t, r, _, w in points])
        play_points = [(t, math.log(p), w) for t, _, p, w in points
                       if isinstance(p, (int, float)) and p > 0]
        play_slope = weighted_slope(play_points) if len(play_points) >= 3 else 0.0
        trend = clamp(-rank_slope * 24.0 / 4.0, -2.0, 2.0) \
            + 0.5 * clamp(play_slope * 24.0, -1.5, 1.5)
        multiplier = math.exp(TREND_GAMMA * trend)
        # Age-out: fade games nearing the 28-day chart limit.
        made = created.get(uid)
        if made is not None:
            age_days = (t_mid - made) / 86400.0
            if age_days >= AGE_HARD_DAYS:
                survival = 0.0
            elif age_days > AGE_SOFT_DAYS:
                survival = (AGE_HARD_DAYS - age_days) / (AGE_HARD_DAYS - AGE_SOFT_DAYS)
            else:
                survival = 1.0
            multiplier *= survival
        if multiplier <= 0.0:
            continue
        for _, rank, _, w in points:
            base = w * multiplier
            for k in range(1, places + 1):
                d = rank - k
                scores[k][uid] = scores[k].get(uid, 0.0) \
                    + base * math.exp(-(d * d) * inv_two_sigma2)

    candidates = sorted(obs)
    n_candidates = max(len(candidates), 1)
    last_chart = []
    if snapshots:
        last_chart = sorted(
            (int(r["universe_id"]) for r in (snapshots[-1].get("up_and_coming") or [])
             if isinstance(r.get("universe_id"), (int, float))),
            key=lambda u: u,
        )
    chart_n = chart_size if chart_size > 0 else len(last_chart)

    probabilities = []
    place_mass = {}
    for k in range(1, places + 1):
        raw = scores[k]
        total = sum(raw.values())
        if total > 0:
            dist = {uid: (1.0 - SHRINK_LAMBDA) * v / total
                    + SHRINK_LAMBDA / n_candidates for uid, v in raw.items()}
            # Candidates with no mass at this place still hold the
            # shrunk uniform share; they will not clear the naming bar.
            for uid in candidates:
                dist.setdefault(uid, SHRINK_LAMBDA / n_candidates)
        else:
            dist = {uid: 1.0 / n_candidates for uid in candidates}
        place_mass[k] = dist
        # Name games while their probability beats the pooled unnamed
        # share they would otherwise receive.
        ranked = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
        named_sum = 0.0
        named = 0
        for uid, p in ranked:
            if named >= MAX_ROWS_PER_PLACE or p <= 0.0:
                break
            unnamed = (1.0 - named_sum) / max(1, chart_n - named)
            if p <= unnamed:
                break
            probabilities.append({"universe_id": uid, "place": k, "p": p})
            named_sum += p
            named += 1
        if named_sum > 0.999999:
            scale = 0.999999 / named_sum
            for row in probabilities:
                if row["place"] == k:
                    row["p"] *= scale

    # Top-10 guess: rank by place-weighted probability mass.
    guess_score = {}
    for k in range(1, places + 1):
        w = places + 1 - k
        for uid, p in place_mass[k].items():
            guess_score[uid] = guess_score.get(uid, 0.0) + w * p
    top10 = [uid for uid, _ in
             sorted(guess_score.items(), key=lambda kv: (-kv[1], kv[0]))[:places]]
    if len(top10) < places:
        for uid in last_chart:
            if uid not in top10:
                top10.append(uid)
            if len(top10) >= places:
                break

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    forecast = {"probabilities": probabilities, "top10": top10[:places]}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_name), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
