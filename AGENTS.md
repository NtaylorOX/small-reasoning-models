# Agent guide

Instructions for AI coding agents working in this repo. `CLAUDE.md` imports this file.

- One project at a time, in roadmap order (see README). Datasets and tools: docs/resources.md. Stop for the owner's review after each step.
- Small PRs, one concern each. `make check` must pass before every push.
- Commits use the owner's identity (`Niall Taylor`, `49034323+NtaylorOX@users.noreply.github.com`).
  No AI attribution anywhere: no `Co-Authored-By` trailers, no "Generated with" lines.
- **Honest results**: say what was run vs only written. Never invent numbers. Every reported number
  comes from a row in `results/` or a linked log.
- Licences, prices and API flags from third-party pages are marked `(unverified)` until checked.
- No secrets in the repo. Keys go in `.env`.
- Python 3.10+, `uv`, type hints, `mypy --strict` on `src/`, `ruff`. Seeds are explicit arguments.
- Tests run offline; mark GPU tests `gpu`, slow ones `slow`.
