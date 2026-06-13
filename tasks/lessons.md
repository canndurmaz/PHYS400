# Lessons (corrections to remember)

## run_pipeline.sh: string-valued flags broke the arg parser

**What I did wrong:** Added `--sampling {random,lhs,sobol,active}` to the pipeline CLI without auditing the shell wrapper's arg-parse heuristic. The wrapper only treated `[0-9]*` tokens as flag values; lowercase string values like `active` fell through to the catch-all and were tagged as element symbols. The user's `--sampling active …` got split into `FLAGS=[--sampling …]` and `ELEMENTS=[active]`, then argparse errored with "expected one argument."

**Rule:** When adding a new CLI flag to `pipeline.py`, also check `run_pipeline.sh`'s arg-parse case statement. The heuristic that worked before (`[0-9]*` = flag value) is incomplete for any string-valued flag. The correct rule is "any token after `--flag` that is not Capitalized is the flag's value" (element symbols are always Capitalized: Al, Cu, Fe, …).

**Also remember:** under `set -euo pipefail`, `${FLAGS[-1]:-}` warns ("bad array subscript") when the array is empty. Guard with `if (( ${#FLAGS[@]} > 0 ))` before indexing.

**Bonus:** smoke-tests should cover the *invocation surface*, not just the library API. A unit test that imports the new module passes is necessary but not sufficient — the user-facing command path is where this kind of bug actually bites.

## Poster columns: `[t]` minipages of tall boxes create huge top whitespace

**What I did wrong:** Laid out the 3-column poster as side-by-side
`\begin{minipage}[t]{...}` each holding a tall `tcolorbox`. `[t]` aligns
minipages by their *first baseline*, not their top edge — so with a tall box as
the first content, the columns misaligned vertically: the tallest column's top
sat ~15% higher than the others, leaving a large empty band under the title. The
user flagged "too much whitespace."

**Rule:** For side-by-side `[t]` minipages that contain a box/graphic as their
first element, put **`\vspace{0pt}` immediately after `\begin{minipage}[t]{..}`**
to force true top alignment. Then balance column *heights* by filling the shorter
ones (bigger figures/diagrams) rather than padding — padding just relocates the
whitespace inside the border.

**Also:** `tcbraster` with `raster equal height=all` is the wrong tool when a row
of boxes can exceed one page — it forces all boxes to the tallest's height and
breaks the row one-box-per-page. Independent minipages are predictable.

**Filling equal-height columns (no bottom blank):** forcing the box height with
`height=… , valign=top` does NOT fill — it parks all slack as one blank gap at
the bottom, and `\vfill` inside does nothing (the content keeps its natural
height). To make columns *fill* top-to-bottom with the slack distributed evenly,
wrap each column body in a **fixed-height stretch minipage**
`\begin{minipage}[t][\colheight][s]{\linewidth}` and put `\vfill` at the section
breaks — `[s]` stretches the `\vfill` glue to fill the height. Identical
`\colheight` + identical title bars ⇒ all borders match top and bottom. (User
correction: "blank spots in bottom; columns should match top and bottom.")

## Commit messages: no Claude/AI attribution

**What I did wrong:** Added the default `Co-Authored-By: Claude …` trailer to a commit; the user rejected it ("no claude affiliation").

**Rule:** Commits and PR bodies in this repo carry no AI-attribution lines of any kind. The user's name is the only author identity.
