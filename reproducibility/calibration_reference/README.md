# NEW calibration equation reference

This directory is a newly supplied, standalone implementation of the revised
paper's source-group tolerance-calibration equations. **It is not the original
experiment pipeline, fitted scoring models, recovered train/validation/
calibration/test splits, or a reproduction of the reported experiments.** No
human annotations, experimental scores, new empirical results, or public-release
commitment are supplied here. The JSON example and all test fixtures are
**SYNTHETIC**, solely to demonstrate the interface and mathematical checks.

## Runtime and use

Python 3.9 or newer and SciPy are required; SciPy is the only direct dependency.

```sh
python -m pip install -r requirements.txt
python calibrate.py example_SYNTHETIC.json --output result_SYNTHETIC.json
python -m unittest -v test_calibrate.py
```

The example requests three families, with an empty third reservoir. It
intentionally demonstrates **global abstention**; an empty or missing requested
family is never silently removed from the Bonferroni denominator. Exit code 0
means the JSON was processed successfully, including statistical abstention.
Malformed input or an I/O failure gives exit code 2. Consumers must inspect
`status`; `shared_cutoff: null` never authorizes deletion.

## Input contract and assumptions

Supply `requested_families`, `epsilon`, `alpha`, `shared_score_scale`, and
`calibration_maxima`. The two probabilities must lie strictly between 0 and 1.
Each requested family maps to one list of clean-group maxima: each entry is
the largest frozen removal score among all legitimate retain candidates in
one group. The code consumes precomputed maxima and does not retrieve pages,
form groups, assess cleanliness, fit a scorer, or split data.

For the population-risk tolerance guarantee, clean calibration groups must be
independently and identically sampled within each family from the same fixed
distribution as future clean groups, with the score and evidence protocol fixed
before calibration. Dependence among candidate rows within a group is allowed;
the independent sampling unit is the whole source group. **Arbitrary joint
exchangeability alone does not establish the stated high-confidence bound on
a fixed population risk.** Independence across families is not needed for the
Bonferroni union bound. Data-driven selection of families, scores, or thresholds
using these calibration outcomes is outside this implementation's contract.

Every family must use the same frozen, compatible score scale and comparison
semantics before taking the maximum of family cutoffs. The scale string merely
records the caller's declaration; the code cannot verify calibration validity,
cleanliness, independence, future distribution stability, or scale compatibility.
Use a fresh deployment-matched reservoir after a material family, language,
retrieval, scorer, or evidence-schema change. Unvalidated transfer has no claimed
guarantee here.

## Formula, ties, and auditable output

Let H be the number of preregistered requested families, including empty ones.
For a family with n clean maxima and integer k in 0,...,n-1, the module evaluates
the exact one-sided Beta bound numerically with SciPy:

`U(k) = scipy.stats.beta.ppf(1 - alpha/H, k + 1, n - k)`.

It chooses the largest k with `U(k) <= epsilon`, using only n and the prespecified
budgets. Its anchor is the `(n-k)`th smallest maximum (one-based rank), and its
cutoff is `math.nextafter(anchor, +infinity)`. The shared cutoff is the maximum
of all family cutoffs. Delete with `score >= shared_cutoff`.

Inputs, anchors, cutoffs, and deployment comparisons must use the same IEEE-754
binary64 representation. The next representable cutoff retains scores tied
with the anchor; ties can make the bound conservative. Do not round the cutoff
or cast it to a lower-precision type, and do not change `>=` to another rule.
The JSON output round-trips Python binary64 values. A nonfinite calibration
score, a nonfinite Beta quantile, an overflowing cutoff, an empty requested
reservoir, or no feasible k causes abstention. Nonfinite future scores must also
be withheld from automatic deletion and reviewed separately.

The output records H, budgets, family sizes, k, order-statistic ranks, anchors,
cutoffs, family risk bounds, observed exceedance counts, abstention reasons,
SciPy version, and (for CLI calls) SHA-256 of the input file. A displayed
`risk_upper_bound` for a `no_feasible_k` family is the diagnostic k=0 bound,
not an authorization to deploy. The family bounds also bound risk at the shared
cutoff under the assumptions, because increasing the cutoff cannot increase
deletion risk. Scientific validity is not verified by a successful program run.

The tests check the n=58/59 single-family boundary, n=79/80 three-family
Bonferroni boundary, the k=1 allowance at n=106, closed-form and binomial-tail
identities, tied maxima, shared-threshold monotonicity, missing-family handling,
nonfinite-score and cutoff-overflow abstention, input rejection, and CLI output.
