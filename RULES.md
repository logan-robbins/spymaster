All rules are mandatory. No exceptions.

## Environment

- **Hardware**: Apple M4 Silicon, 128GB RAM
- **Python tooling**: latest python 3.12, `uv` exclusively. No `pip`, no raw `python`, no manual `venv`.
  - Run scripts: `uv run script.py`
  - Run tests: `uv run pytest`
  - Add packages: `uv add package`
- **Virtual environment**: `.venv` in the project root or child workspace. This is the single source of truth for dependencies.

## Workflow

Follow this sequence for every task:

1. **Discover** — Search the codebase for existing patterns, utilities, or similar implementations before creating anything new. Extend existing code when possible.
2. **Plan** — Create a living task document with numbered items. Update it as work progresses.
3. **Implement** — Write one canonical solution. No fallback paths, no optional modes. If prerequisites are unmet, fail fast with a clear error.
4. **Verify** — Run `uv run <command>` locally to confirm the implementation works.
5. **Update** — Mark completed items in the task document with minimal status annotations.

## Code Standards

- **One implementation**: No duplicate files, no parallel code paths, no "v2" copies. Delete what you replace.
- **Clean as you go**: Remove outdated comments, dead functions, and orphaned files.
- **Fail fast**: Return clear, specific errors when prerequisites are missing.
- **Raw data is immutable**: Never modify, delete, or alter raw data unless specifically instructed. All other data may be regenerated as needed.
- **Hard path only**: We never skip tasks, change libraries or approach strategies because they are "hard", we spend unlimited time and money on the right way.

## Execution Rules

- **Long-running commands**: Always use `nohup` with verbose logging. Poll output in 15-second intervals and terminate if stalled or erroring.
- **README.md**: After completing work, update `README.md` to reflect the current system state. These documents are written for AI/LLM consumption — include specific commands, configuration, and key information needed to launch, run, and debug the system.

## Output Rules

- **No artifact files**: Never create summary markdown files, "final guide" documents, or documentation artifacts unless the user explicitly requests one.
- **No unsolicited reading**: Do not read markdown documents unless the user specifically requests it or the workflow requires it (e.g., README.md, links in README.md).
- **Summaries in responses only**: End substantive responses with a brief summary in your reply text. No code blocks in summaries. Skip summaries for simple questions.
