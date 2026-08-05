# Working notes

## Rules

- Change only what was asked. Leave unrelated code, formatting, files and
  wording alone, even when something nearby looks improvable — mention it
  instead.
- Before returning a result, check it. Re-run what was changed and verify every
  number and claim against the source file or a computation, rather than
  assuming it followed from the previous step.
- Explain short. Plain words, no long write-ups unless asked for detail.
- Use only this project's skills, in `~/dev/bird-behavior/.claude/skills`. Other
  repos in this VS Code workspace (hedge-seg, ...) have same-named skills;
  never run theirs.
- Commit messages: never mention Claude or Claude Code (no attribution, no
  Co-Authored-By line, no tool names).
- Always update the related documents in the same change (see Documentation).
- Never run `git commit`. Prepare the change and the message; the user commits.
- Always end a change with a ready-to-paste commit message, without being asked.
  Give it in a copyable block: a ~50-character imperative subject, then a body
  saying what changed and why, wrapped at 72 characters. Say so explicitly if
  the change is not worth committing on its own.
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
- Keep `docs/lesson_learned.md` and `docs/experiments_log.md` short, and put new
  entries at the top.
- Always call python by its full path: `/home/fatemeh/miniconda3/envs/bird/bin/python`.
  Not bare `python`, not `conda run -n bird`. VS Code puts one env's `bin` at the
  front of `PATH` for every tool call, and conda does not know it is there, so
  `conda run` and `conda activate` can silently run the wrong env. A full path
  cannot be shadowed.

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

- `docs/description.md`: overview of the data, models, and every script. Keep it
  polished.
- `docs/lesson_learned.md`: curated lessons that generalize, with the reason
  behind each.
- `docs/experiments_log.md`: terse per-run notebook. Held to a lower bar than
  the other two.
- `presentation/presentation.md`: the talk. Highlights only, short and
  itemised. Where `lesson_learned.md` gives the reasoning, this gives the conclusion
  in one line. Keep a technical name in italic parentheses after the plain-language
  version, so the audience follows and a specialist can still place it.
- `presentation/README.md`: which figures the talk uses, what each set shows,
  and the exact steps to remake them. No findings here, only mechanics.

After a change: update the script's top docstring and `docs/description.md` for
new or changed behavior; add to `docs/lesson_learned.md` for a bug, surprise, or
design decision (with the reason); add to `docs/experiments_log.md` for a
finished or planned run.
