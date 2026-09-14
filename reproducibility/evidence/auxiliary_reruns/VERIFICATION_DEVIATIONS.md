# Auxiliary verification deviation record

The initial verification command read new prediction CSVs with pandas'
default numeric parser. It exited with an assertion failure in the saved
cross-lingual calibration decisions: one mismatched element among 432 rows.
The error was observed in tool stderr; stderr had not been redirected to a
raw file. The original empty `independent_auxiliary_20260914/verification.json`
is preserved. This document records that failure instead of pretending it
was captured as a raw execution log.

Using `float_precision="round_trip"` recovers the original binary floats
from their CSV representation. The independent model inference, threshold
and decision checks then passed. A subsequent unchanged-model export run
added `scores.npz`; its results, prediction CSVs and frozen model arrays
were byte-identical to the first run. The final verifier also checks binary
scores against round-trip parsed CSV values. No model, split, calibration
setting or reported count was changed to resolve the parser issue.

A further standalone execution without the primary-state cache rebuilt the
primary model from the fixed training CSVs. Its results, prediction CSVs,
binary score arrays, frozen model weights and primary MCQ calibration JSON
are byte-identical to the cache-assisted run. See the final output's
`standalone_verification.json`.
