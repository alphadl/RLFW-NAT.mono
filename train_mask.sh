#!/usr/bin/env bash
# Train Mask-Predict on monolingual KD binarized data.
# Env: SRC, TGT, DATA (path to binarized data), SAVE_DIR, [GPU].
set -e
SRC="${SRC:-en}"
TGT="${TGT:-de}"
DATA="${DATA:?Set DATA=path/to/databin}"
SAVE_DIR="${SAVE_DIR:?Set SAVE_DIR=path/to/save}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME="mask_$(basename "$DATA")"
SAVE_DIR="${SAVE_DIR%/}"
mkdir -p "$SAVE_DIR"

pip install -e "$ROOT/fairseq_mask/"

echo ">> training Mask-Predict on $DATA -> $SAVE_DIR"
fairseq-train "$DATA" \
  --save-dir "$SAVE_DIR" \
  --ddp-backend=no_c10d --fp16 \
  --task translation_lev \
  --criterion nat_loss \
  --arch cmlm_transformer \
  --label-smoothing 0.1 \
  --attention-dropout 0.0 \
  --activation-dropout 0.0 \
  --dropout 0.2 \
  --noise random_mask \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 1e-07 --max-lr 1e-3 --lr-scheduler cosine \
  --warmup-init-lr 1e-07 --warmup-updates 10000 --lr-shrink 1 --lr-period-updates 60000 \
  --max-update 70000 \
  --weight-decay 0.0 --clip-norm 0.1 \
  --max-tokens 20000 --update-freq 3 \
  --decoder-learned-pos \
  --encoder-learned-pos \
  --apply-bert-init \
  --no-progress-bar --log-format 'simple' --log-interval 100 \
  --fixed-validation-seed 7 \
  --seed 1 \
  --save-interval-updates 2000 \
  --keep-last-epochs 0 \
  --fp16-scale-tolerance 0.1

# For small data (e.g. en-ro) you may use:
#   --attention-dropout 0.3 --activation-dropout 0.3 --dropout 0.3 \
#   --share-all-embeddings \
#   --warmup-updates 4000 --lr-period-updates 21000 --max-update 25000 \
#   --weight-decay 0.0001
