# Retained execution logs

These logs record the new complete constructed run, an earlier diagnostic fit, external reruns, auxiliary replay, the overwrite guard, and a rejected six-page build. The initial diagnostic is not the current reported run. The final build is recorded one level above in `build.log`.

The first artifact-marker invocation failed because it ran from the wrong directory; the required marker then succeeded once from the PDF skill directory. Two initial manuscript layouts exceeded five pages. The preserved six-page log records the second failure; the first failure was captured only in the task output. The final layout shortened added explanations and the new caption without changing font size, page margins, or the original figure width.

An initial CSV-based numerical replay used default pandas float parsing and failed bit-level score checks. The checker was corrected to use round-trip parsing and binary exported score arrays. This did not change source labels, fitted model parameters, or the reported run. Model serialization caches were not included in the release; numerical weights and features support portable inference checks.

The initial delivery-manifest preparation referenced a nonexistent combined auxiliary `results.json` and stopped before writing the root manifest. The correct separate `web_pilot_results.json` and `xling_results.json` paths were then used, and integrity checks were rerun successfully.
