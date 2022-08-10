#!/usr/bin/env bash
# Decode with Levenshtein NAT. DATA=databin, CHECKPOINT=dir with .pt, SUBSET=valid|test.
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATA="${DATA:?Set DATA=path/to/databin}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=path/to/checkpoint_dir}"
SUBSET="${SUBSET:-valid}"
CKPT="${CKPT:-checkpoint_best.pt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$CHECKPOINT/gen}"
mkdir -p "$OUT"

echo ">>> validating $SUBSET"
python "$ROOT/fairseq_lev/fairseq_cli/generate.py" "$DATA" \
  --path "$CHECKPOINT/$CKPT" -s $SRC -t $TGT \
  --gen-subset "$SUBSET" --task translation_lev \
  --iter-decode-max-iter 10 --iter-decode-eos-penalty 0 \
  --iter-decode-with-beam 1 --remove-bpe --batch-size 64 --print-step \
  > "$OUT/${SUBSET}.out" 2> "$OUT/${SUBSET}.log"
grep ^H "$OUT/${SUBSET}.out" | cut -f3- > "$OUT/${SUBSET}.hyp"
echo "Hyp: $OUT/${SUBSET}.hyp"
