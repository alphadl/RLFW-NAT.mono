#!/usr/bin/env bash
# Test-set decode + BLEU (Levenshtein). Set DATA, CHECKPOINT, REF.
set -e
SUBSET=test
export SUBSET
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT/eval_lev.sh"
OUT="${OUT:-${CHECKPOINT:?}/gen}"
[ -n "$REF" ] && [ -f "$REF" ] && sacrebleu "$REF" -i "$OUT/test.hyp" -b
