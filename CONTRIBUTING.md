# Contributing

Thanks for helping make Limes Labs technical work reproducible and honest.

## First Local Check

Run:

```bash
./scripts/smoke_test.sh
python3 -m unittest
```

If either command fails, open an issue or PR with the failing command, platform,
Python version, and last 30 lines of output.

## What Good Contributions Look Like

- Small, reviewable patches.
- Clear configs with explicit dataset, device, and iteration counts.
- Legal data provenance in plain language.
- Eval artifacts that can be copied into EuroBench or a model card.
- Negative results documented in `EXPERIMENTS.md`.
- No capability claims without a reproducible artifact.

## ML Engineering Guidelines

- Keep dependencies minimal.
- Prefer tiny public or Limes-authored data for default tests.
- Make large downloads optional and documented.
- Do not commit checkpoints, generated binaries, or private data.
- Include the command that produced every metric.
- Use `config/train_smoke.py` when changing onboarding behavior.

## Pull Request Checklist

- [ ] `python3 -m unittest` passes.
- [ ] Smoke path still writes `out-smoke/eval.json`.
- [ ] README or `EXPERIMENTS.md` updated for user-visible workflow changes.
- [ ] New data has a license/provenance note.
- [ ] Claims are limited to what the artifacts show.

## Suggested First Issues

- Add JSON schema validation for `eval_perplexity.py` outputs.
- Add a tiny EuroBench adapter that reads `out-smoke/eval.json`.
- Add checksums for prepared dataset binaries.
- Add a model-card example generated from the smoke artifact.
