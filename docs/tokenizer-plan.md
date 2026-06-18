# Multilingual European Tokenizer Plan

The current repo uses a tiny character-level dataset for onboarding. That is
fine for smoke tests, but future European experiments need a documented path
toward BPE or SentencePiece-style tokenization over multilingual data.

## Scope

The first tokenizer milestone should stay dependency-light. No heavy tokenizer
dependency should be added until there is a tiny fixture corpus, a measured
baseline, and tests that prove the dependency is worth carrying.

Target language coverage for the first fixture:

- Italian, including accented legal and administrative text.
- German, including compounds and umlauts.
- French and Spanish, including contractions and diacritics.
- Polish, Czech, Romanian, Greek, and at least one smaller regional or
  low-resource European language when licensing is clear.
- Mixed EU institution text with currency, dates, citations, and article
  references.

## Candidate Path

1. Keep the committed starter corpus as character-level input.
2. Add a small UTF-8 fixture with multilingual public-domain or Limes-authored
   text.
3. Implement tokenizer-evaluation helpers before training a tokenizer.
4. Compare character-level, byte-level BPE, and SentencePiece-compatible output
   formats.
5. Choose a production dependency only when it improves compression,
   round-trip safety, and training ergonomics enough to justify it.

## Acceptance Tests

A future BPE or SentencePiece path should pass these checks before becoming the
default:

- Round-trip: decode(encode(text)) equals the original UTF-8 text for every
  fixture line.
- Diacritics: accented Latin text, Greek text, and Polish/Czech characters
  survive round-trip without normalization surprises.
- Boundary stability: citations, section numbers, dates, URLs, and euro amounts
  are not silently dropped.
- Bits per byte: tokenizer comparisons report validation loss as bits per byte,
  not only token loss.
- Fixture licensing: every source file has a license note or is Limes-authored.
- No heavy tokenizer dependency: pure-Python tests keep running without optional
  native tokenizer packages.
- Determinism: the same seed and fixture create the same vocabulary artifacts.
- Budget accounting: vocabulary size, artifact bytes, training examples, raw
  bytes, and elapsed seconds are written to JSON.

## Non-Goals

- No scraped multilingual web corpus in this repo yet.
- No claim that tokenization alone makes a multilingual benchmark.
- No dependency on private data.
- No change to `scripts/smoke_test.sh` until the tokenizer path is optional and
  fast on CPU.
