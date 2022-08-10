#!/usr/bin/env bash
# Test-set decode + BLEU (Mask-Predict). Set DATA, CHECKPOINT, REF (reference file).
set -e
SUBSET=test
export SUBSET
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT/eval_mask.sh"
OUT="${OUT:-${CHECKPOINT:?}/gen}"
[ -n "$REF" ] && [ -f "$REF" ] && sacrebleu "$REF" -i "$OUT/test.hyp" -b
