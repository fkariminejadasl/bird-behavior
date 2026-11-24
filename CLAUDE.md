# Working notes

## Start of a session

- Read `docs/status.md`, `docs/descriptions.md` (the Quick reference first),
  `docs/lesson_learned.md` and `docs/experiments_log.md` before proposing or
  running anything. They are not loaded automatically, and they hold what has
  already been tried, measured and ruled out.
- Answer from this repository only. The other repos in this VS Code workspace
  (hedge-seg, ...) are not context unless I say so. That includes skills: use
  only this project's, in `~/dev/bird-behavior/.claude/skills`, because the
  other repos have same-named ones.

## How to work

- Change only what was asked. Leave unrelated code, formatting, files and
  wording alone, even when something nearby looks improvable — mention it
  instead.
- Before returning a result, check it. Re-run what was changed and verify every
  number and claim against the source file or a computation, not against an
  earlier step or another doc. Say so when a claim is a guess.
- Say which files were inspected before answering a question about the repo.
  Give a full clickable path, and name the exact function or class.
- Ask before running an experiment (a training run): it costs GPU time and
  hours. Ask before installing a package; once approved, add it to
  `pyproject.toml` with a short comment saying what it is for.
- Never overwrite or delete a file outside the repo without asking first. Write
  new outputs under a new name.
- Call python and the formatters by full path,
  `/home/fatemeh/miniconda3/envs/bird/bin/python` and so on, not `python` or
  `conda run -n bird`: VS Code puts one env's `bin` first on `PATH`, unknown to
  conda, so they can silently run the wrong env.
- No `argparse`. A script's config is a dict wrapped with OmegaConf, under
  `if __name__ == "__main__":`, edited in place to change a run.

## How to write

- Explain short, in chat and in the docs. Plain words, short sentences, no
  clever phrasing. Spell out an acronym the first time (VeDBA, ODBA). If a
  sentence needs re-reading, rewrite it. Asked what something is, give the
  definition, one real example and the counts.
- Use the words of the Terms table in `docs/descriptions.md` (record, fix,
  burst, bout, segment, window); do not coin one such as "recording".
- A title states the result in plain words, with its runs: "Three magnitude
  channels win back half the accuracy that rotation costs (exp197)", not a
  slogan like "Give the model an invariant". A bullet starts with its result
  too, not a bold catchphrase. Take the result from the numbers, not from the
  story told about them.
- Keep every document short: `status.md`, `descriptions.md`,
  `lesson_learned.md`, `experiments_log.md`, `presentation.md`. Prefer a table
  or a bullet to a paragraph, and when adding, look for something to cut. The
  Quick reference in `descriptions.md` is the shortest of all — an index, one
  line per entry. New `experiments_log` entries go at the top; a new lesson goes
  at the top of its group in `lesson_learned.md`.

## Numbers

- Name the unit of every count and table column (records, fixes, segments) and
  the set it counts, so two numbers are never compared across them.
- Name the metric of every score and what it was scored on: accuracy in %, or
  F1 on 0-1 (per class or pooled, plain or class-balanced), over which items
  (segments or records), which split, and which checkpoint: the best epoch
  (`<exp>_best.pth`) or the last (`<exp>_4000.pth`). A bare 0.91 cannot be
  compared with a paper's number. In `status.md` and on a slide, put the metric
  and the checkpoint in the table header or the line above the table.
- Every number in a doc needs code in the repo that prints it: in `exps/`, or a
  check in `exps/recompute_doc_numbers.py`. A number whose code is gone cannot
  be rechecked: drop it, or say there is no script.
- **Name that script wherever the number appears**: in the sentence in
  `status.md`, `lesson_learned.md` and `descriptions.md`; in italic parentheses
  at the end of the line in `presentation.md`, for example
  `*(exp 196-197, exps/eval_unlabeled.py)*`. The script is the one that printed
  the number, not the one that trained the model.

## Commits

- Never run `git commit`. Prepare the change and the message; the user commits.
- Always end a change with a ready-to-paste commit message, without being asked,
  covering everything still uncommitted (`git status`). Say so if the change is
  not worth committing.
- **Keep it short**: a ~50-character imperative subject and a body of a few
  lines wrapped at 72. Say what changed and why. Leave out numbers and reasoning
  — `experiments_log.md` and `lesson_learned.md` already hold them. Never
  mention Claude or Claude Code (no attribution, no Co-Authored-By line, no tool
  names).
- Give the full commands in a copyable block: `git add` with the files named,
  from `git status --short` (leave out paths already staged, such as a
  `git mv`), then `git commit -m "..."`. **So the message may not contain any of
  these four characters:**

  | character | why it breaks |
  |---|---|
  | `"` | closes the quote early |
  | `'` | breaks the shell if I quote with it instead |
  | `` ` `` | runs as a command inside double quotes |
  | `$` | expands as a variable inside double quotes |

  Write `model=BirdModelSmallDilated`, not `` `model="..."` ``. Say: the 0.05
  threshold, without quotes. Blank lines between paragraphs are fine inside the
  quotes.

## Format and check

After each change:

```bash
B=/home/fatemeh/miniconda3/envs/bird/bin
for i in behavior exps scripts app; do echo $i; $B/black $i -l 88; $B/isort $i --profile black; $B/pyflakes $i; done
$B/mdformat --number --exclude docs/cheatsheets/clustering.md docs/
git diff -U0 -- 'docs/*.md' | grep '^+[^+|]' | awk 'length > 81'
```

- `mdformat` and `mdformat-gfm` are in the `test` extra of `pyproject.toml`.
  Without `--number`, ordered lists collapse to all `1.`; with it, a list of 10
  or more items is zero-padded (`01.`), so keep them under 10. Without the
  exclude, the LaTeX math of `docs/cheatsheets/clustering.md` and its `---`
  rules are rewritten into broken text and rows of underscores.
- A backslash escape added by mdformat means ambiguous markdown: use backticks,
  or reflow so that no line starts with a number and a period.
- mdformat does not rewrap paragraphs; the last command prints added lines over
  80 characters to rewrap (the Quick reference keeps one line per entry). An
  edit that ends mid-line joins the rest of the old line onto it.
- Never run mdformat on `presentation/`. A Marp slide must fit one page; Marp
  silently cuts the bottom. Keep about 12 lines of body text: a line over about
  72 visible characters counts twice, and a title over about 48 characters
  takes two lines. Estimates only; Marp is not rendered here, so say which
  slides to export and check.

## Workspace

- Scratch files: `/home/fatemeh/Downloads/bird/claude`, not `/tmp` (wiped on
  shutdown). Clean up whatever is not worth keeping.
- Images worth keeping: `/home/fatemeh/Downloads/bird/screenshots`.
- New datasets: one folder per source under `/home/fatemeh/Downloads/bird/data`,
  kept as released. Note the links in `docs/descriptions.md`.
- Settings for this project live in `~/dev/bird-behavior/.claude` (for example
  `settings.json`), never another repo's `.claude` in this VS Code workspace.

## Reporting a result

- After a run, say what ran, what changed outside the repo (and whether anything
  was overwritten), and the commands used, naming the config keys that changed.
- Experiment numbers are global and never reused: a rerun replaces the
  checkpoint, metrics and tensorboard run of the old one.
  `scripts/batch_train_supervised.py` refuses to start on a number that already
  has results, unless `overwrite` is set. The highest number in use:

  ```bash
  ls /home/fatemeh/Downloads/bird/results | grep -oP "^\d+(?=_best\.pth$)" | sort -n | tail -1
  ```

## Documentation

Update the related documents in the same change:

- `docs/status.md`: where the project stands. The working models, how the
  numbers are measured, the results tables, Done, TODO and Dropped.
- `docs/descriptions.md`: overview of the data, models and every script. Keep it
  polished.
- `docs/lesson_learned.md`: curated lessons that generalize, with the reason.
- `docs/experiments_log.md`: terse per-run notebook, held to a lower bar.
- `docs/methods.md`: how raw data becomes the model input (the database and
  Ornitela rules), the model and the training.
- `docs/cheatsheets/`: general how-to notes (pandas, OmegaConf, database
  queries, clustering metrics, loading large datasets). No findings here.
- `presentation/presentation.md`: the talk. Highlights only, short and itemised:
  the conclusion in one line, where `lesson_learned.md` gives the reasoning. A
  technical name goes in italic parentheses after the plain-language version,
  so a specialist can still place it.
- `presentation/update.md`: the meeting updates, one slide per meeting, newest
  first. The user writes it. Do not add to it, reorder it or fill it in; only
  correct the wording and nothing else.
- `presentation/README.md`: which figures the talk uses, what each shows, and
  the exact steps to remake them. No findings here, only mechanics.

Each result is written in full in one place only: the details of a run in
`experiments_log.md`, the results tables in `status.md`. Other docs point
there, so two copies cannot drift apart. The talk is the exception: it repeats
numbers, each with its measure and its script.

After a change: the script's top docstring and `descriptions.md` for new or
changed behavior; `lesson_learned.md` for a bug, surprise or design decision,
with the reason; `experiments_log.md` for a finished or planned run, and the
results tables, Done and TODO in `status.md`. When a number the talk quotes
changes, update `presentation.md` in the same change — it goes stale silently,
because nothing breaks when it is wrong.
