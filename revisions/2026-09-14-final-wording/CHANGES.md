# Four minimal manuscript changes

Status: approved by the author for publication to main on 2026-09-14 UTC. The four edits below are the complete manuscript change from v2.

Baseline release: `iccc2026-repro-20260914-v2`.

## Scorer naming

Before:

```latex
similarity-plus-chronology and full-evidence logistic models
```

After:

```latex
no-constraint (similarity-plus-chronology) and full-evidence logistic models
```

## MCQ calibration bound

Before:

```latex
the associated 4.67\% calibration bound
```

After:

```latex
the MCQ calibration bound of 4.67\%
```

## Figure interpretation

Before:

```latex
Figure~\ref{fig:tradeoff} separates ranking from policy performance: similar rankings can yield different clean-retention tradeoffs, so
```

After:

```latex
Figure~\ref{fig:tradeoff} compares calibrated operating points;
```

## Text-deduplication limitation

Before:

```latex
Some retain documents pair with multiple source items, so identical text may occur in different partitions without splitting a group.
```

After:

```latex
Some retain documents pair with multiple source items, so identical text may occur in different partitions without splitting a group. The current rerun does not assess robustness under strict cross-partition text deduplication.
```

## Scope and validation

Exactly four authorized source replacements. The abstract, author block, reported numbers, citations, figures, equations, and formatting commands are unchanged. The compiled PDF has five pages. The experiment code, data, and frozen results remain those of the v2 release; this wording-only revision introduces no new experiment.
