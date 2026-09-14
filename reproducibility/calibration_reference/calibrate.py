#!/usr/bin/env python3
"""NEW equation reference; not the original SafeDecontam experiment pipeline.

Confidence is over i.i.d. clean-group calibration samples from each family's
fixed deployment distribution. Arbitrary exchangeability alone is insufficient
for this population-risk tolerance claim. See README.md for the full contract.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import scipy
from scipy.stats import beta


def upper_bound(n, k, family_alpha):
    """Beta^{-1}(1-family_alpha; k+1, n-k), with 0 <= k < n."""
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k < n:
        raise ValueError("k must be an integer in [0, n-1]")
    if not 0 < family_alpha < 1:
        raise ValueError("family_alpha must be in (0, 1)")
    return float(beta.ppf(1.0 - family_alpha, k + 1, n - k))


def _family_result(name, maxima, epsilon, family_alpha):
    if not isinstance(maxima, list):
        raise ValueError(f"calibration_maxima[{name!r}] must be a list")
    result = {
        "family": name, "n_clean_groups": len(maxima),
        "family_alpha": family_alpha, "status": "abstain",
        "reason": None, "allowed_exceedances_k": None,
        "order_statistic_rank_1_based": None, "anchor_maximum": None,
        "family_cutoff": None, "risk_upper_bound": None,
        "observed_exceedances_at_family_cutoff": None,
    }
    if not maxima:
        result["reason"] = "empty_or_missing_requested_family"
        return result
    scores = []
    for value in maxima:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"calibration_maxima[{name!r}] contains a nonnumeric score")
        try:
            score = float(value)
        except OverflowError:
            score = math.inf
        if not math.isfinite(score):
            result["reason"] = "nonfinite_calibration_score"
            return result
        scores.append(score)
    scores.sort()
    n = len(scores)
    # k is chosen solely from (n, epsilon, family_alpha), never from scores.
    lower_bound = upper_bound(n, 0, family_alpha)
    if not math.isfinite(lower_bound):
        result["reason"] = "nonfinite_beta_quantile"
        return result
    if lower_bound > epsilon:
        result["reason"] = "no_feasible_k"
        result["risk_upper_bound"] = lower_bound
        return result
    # The Beta upper bound is nondecreasing in k. Find the largest feasible k.
    lo, hi = 0, n - 1
    while lo < hi:
        middle = (lo + hi + 1) // 2
        bound = upper_bound(n, middle, family_alpha)
        if not math.isfinite(bound):
            result["reason"] = "nonfinite_beta_quantile"
            return result
        if bound <= epsilon:
            lo = middle
        else:
            hi = middle - 1
    k = lo
    bound = upper_bound(n, k, family_alpha)
    anchor = scores[n - k - 1]
    cutoff = math.nextafter(anchor, math.inf)
    result.update({
        "allowed_exceedances_k": k,
        "order_statistic_rank_1_based": n - k,
        "anchor_maximum": anchor,
        "risk_upper_bound": bound,
    })
    if not math.isfinite(cutoff):
        result["reason"] = "no_finite_cutoff_above_anchor"
        return result
    result.update({
        "status": "calibrated", "reason": None,
        "family_cutoff": cutoff,
        "observed_exceedances_at_family_cutoff": sum(q >= cutoff for q in scores),
    })
    return result


def calibrate(spec):
    """Consume preregistered family names and one clean maximum per group.

    Missing requested families remain in H and cause global abstention.
    A score scale identifier is a caller declaration, not a validation of scale
    compatibility, cleanliness, independence, or distributional stability.
    """
    if not isinstance(spec, dict):
        raise ValueError("input must be a JSON object")
    families = spec.get("requested_families")
    if (not isinstance(families, list) or not families
            or any(not isinstance(h, str) or not h.strip() for h in families)
            or len(families) != len(set(families))):
        raise ValueError("requested_families must be a nonempty list of unique names")
    maxima = spec.get("calibration_maxima")
    if not isinstance(maxima, dict):
        raise ValueError("calibration_maxima must be an object mapping families to lists")
    if set(maxima) - set(families):
        raise ValueError("calibration_maxima contains unrequested families")
    scale = spec.get("shared_score_scale")
    if not isinstance(scale, str) or not scale.strip():
        raise ValueError("shared_score_scale must identify the same frozen scale for all families")
    parameters = {}
    for key in ("epsilon", "alpha"):
        value = spec.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 < value < 1):
            raise ValueError(f"{key} must be a finite number in (0, 1)")
        parameters[key] = float(value)
    epsilon, alpha = parameters["epsilon"], parameters["alpha"]
    family_alpha = alpha / len(families)
    if not 0.0 < 1.0 - family_alpha < 1.0:
        raise ValueError("alpha/H is outside the supported floating-point quantile resolution")
    results = [_family_result(h, maxima.get(h, []), epsilon, family_alpha)
               for h in families]
    success = all(r["status"] == "calibrated" for r in results)
    shared_cutoff = max(r["family_cutoff"] for r in results) if success else None
    return {
        "implementation": "NEW equation reference, not original experimental code",
        "scipy_version": scipy.__version__,
        "status": "calibrated" if success else "abstain",
        "reason": None if success else "at_least_one_requested_family_abstained",
        "epsilon": epsilon, "alpha": alpha, "H": len(families),
        "requested_families": families,
        "shared_score_scale": scale,
        "deletion_rule": "remove iff finite score >= shared_cutoff; otherwise abstain",
        "shared_cutoff": shared_cutoff,
        "assumptions_verified_by_code": False,
        "families": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with preregistered families and clean maxima")
    parser.add_argument("--output", type=Path, help="write auditable JSON here instead of stdout")
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        result = calibrate(json.loads(raw))
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Input/output error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
