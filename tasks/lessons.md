# Lessons (corrections to remember)

## run_pipeline.sh: string-valued flags broke the arg parser

**What I did wrong:** Added `--sampling {random,lhs,sobol,active}` to the pipeline CLI without auditing the shell wrapper's arg-parse heuristic. The wrapper only treated `[0-9]*` tokens as flag values; lowercase string values like `active` fell through to the catch-all and were tagged as element symbols. The user's `--sampling active …` got split into `FLAGS=[--sampling …]` and `ELEMENTS=[active]`, then argparse errored with "expected one argument."

**Rule:** When adding a new CLI flag to `pipeline.py`, also check `run_pipeline.sh`'s arg-parse case statement. The heuristic that worked before (`[0-9]*` = flag value) is incomplete for any string-valued flag. The correct rule is "any token after `--flag` that is not Capitalized is the flag's value" (element symbols are always Capitalized: Al, Cu, Fe, …).

**Also remember:** under `set -euo pipefail`, `${FLAGS[-1]:-}` warns ("bad array subscript") when the array is empty. Guard with `if (( ${#FLAGS[@]} > 0 ))` before indexing.

**Bonus:** smoke-tests should cover the *invocation surface*, not just the library API. A unit test that imports the new module passes is necessary but not sufficient — the user-facing command path is where this kind of bug actually bites.

## Commit messages: no Claude/AI attribution

**What I did wrong:** Added the default `Co-Authored-By: Claude …` trailer to a commit; the user rejected it ("no claude affiliation").

**Rule:** Commits and PR bodies in this repo carry no AI-attribution lines of any kind. The user's name is the only author identity.
