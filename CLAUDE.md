# Working notes

## Rules

- At the start of a session, read `docs/descriptions.md` (the Quick reference
  section first), `docs/lesson_learned.md` and `docs/experiments_log.md` before
  proposing or running anything. They are not loaded automatically, and they
  hold what has already been tried, measured and ruled out.
- Change only what was asked. Leave unrelated code, formatting, files and
  wording alone, even when something nearby looks improvable — mention it
  instead.
- Before returning a result, check it. Re-run what was changed and verify every
  number and claim against the source file or a computation, rather than
  assuming it followed from the previous step.
- Explain short, in chat and in the docs. Plain words, short sentences, no
  clever phrasing. Spell out an acronym the first time (VeDBA, ODBA).
  If a sentence needs re-reading, rewrite it.
- Use only this project's skills, in `~/dev/bird-behavior/.claude/skills`. Other
  repos in this VS Code workspace (hedge-seg, ...) have same-named skills;
  never run theirs.
- Commit messages: never mention Claude or Claude Code (no attribution, no
  Co-Authored-By line, no tool names).
- A commit message must cover everything still uncommitted (`git status`), not
  only the latest change. Check what is pending before writing it.
- Always update the related documents in the same change (see Documentation).
- Never run `git commit`. Prepare the change and the message; the user commits.
- Always end a change with a ready-to-paste commit message, without being asked,
  in a copyable block. **Keep it short**: a ~50-character imperative subject and
  a body of a few lines wrapped at 72. Say what changed and why. Leave out
  numbers and reasoning — `experiments_log.md` and `lesson_learned.md` already
  hold them. Say so if the change is not worth committing.
- Ask for permission before running an experiment (training run). Running one
  costs GPU time and hours.
- When reporting a path, give a full clickable path so it is easy to check.
- Ask before installing a package. Once approved, install it and add it to
  `pyproject.toml` with a short comment saying what it is for.
- No `argparse`. Configure scripts the way the rest of the repo does: a config
  dict wrapped with OmegaConf, defined under `if __name__ == "__main__":` and
  edited in place to change a run.
- After each change, format and check:

  ```bash
  for i in behavior exps scripts app; do echo $i; black $i -l 88; isort $i --profile black; pyflakes $i; done
  ```
- Keep every document short: `descriptions.md`, `lesson_learned.md`,
  `experiments_log.md`, `presentation.md`, `update.md`. Prefer a table or a
  bullet to a paragraph, and when adding, look for something to cut. The Quick reference in
  `descriptions.md` is the shortest of all — an index, one line per entry, detail
  further down. New `lesson_learned` and `experiments_log` entries go at the top.
- Always call python by its full path: `/home/fatemeh/miniconda3/envs/bird/bin/python`.
  Not bare `python`, not `conda run -n bird`. VS Code puts one env's `bin` at the
  front of `PATH` for every tool call, and conda does not know it is there, so
  `conda run` and `conda activate` can silently run the wrong env. A full path
  cannot be shadowed.
- **Name that script wherever the number appears.** In `lesson_learned.md` and
  `descriptions.md` write the path in the sentence; in `presentation.md` put it
  in italic parentheses at the end of the line, for example
  `*(exp 196-197, exps/eval_unlabeled.py)*`.

## Workspace

- Scratch/temporary files: use `/home/fatemeh/Downloads/bird/claude`, not `/tmp`
  (which is wiped on shutdown). Clean up whatever is not worth keeping.
- Images worth keeping (figures for a presentation, screenshots): save to
  `/home/fatemeh/Downloads/bird/screenshots`.
- Settings/config for this project live in `~/dev/bird-behavior/.claude` (for
  example `settings.json`), never another repo's `.claude` in this VS Code
  workspace.

## Documentation

Keep these current whenever the code they describe changes:

- `docs/descriptions.md`: overview of the data, models, and every script. Keep it
  polished.
- `docs/lesson_learned.md`: curated lessons that generalize, with the reason
  behind each.
- `docs/experiments_log.md`: terse per-run notebook. Held to a lower bar than
  the other two.
- `presentation/presentation.md`: the talk. Highlights only, short and
  itemised. Where `lesson_learned.md` gives the reasoning, this gives the conclusion
  in one line. Keep a technical name in italic parentheses after the plain-language
  version, so the audience follows and a specialist can still place it.
- `presentation/update.md`: the meeting updates, a second Marp deck. One
  slide per meeting, newest first, in short bullets: what was
  done, what is next. Where `presentation.md` builds one
  coherent story, this is the running progress report and may be rough.
- `presentation/README.md`: which figures the talk uses, what each set shows,
  and the exact steps to remake them. No findings here, only mechanics.

After a change: update the script's top docstring and `docs/descriptions.md` for
new or changed behavior; add to `docs/lesson_learned.md` for a bug, surprise, or
design decision (with the reason); add to `docs/experiments_log.md` for a
finished or planned run. When a number the talk quotes changes, update
`presentation/presentation.md` in the same change — it goes stale silently,
because nothing breaks when it is wrong. Add anything finished, started or
dropped since the last meeting to `presentation/update.md`, as a bullet on
the newest slide; start a new slide at the top once that meeting has passed.
