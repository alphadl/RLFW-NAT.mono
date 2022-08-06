#! /usr/bin/bash
export MKL_SERVICE_FORCE_INTEL=1;
export MKL_THREADING_LAYER=GNU;

ps aux|grep /root/miniconda2/envs/py3.7/bin/python|awk '{print $2}'|xargs kill -9

pip install -e ./fairseq_mask/

set -e

s=$SRC
t=$TGT

databin=${databin}
task=mask_${databin}

if [ ! -d ./checkpoint/${s}${t}/${task} ]; then
  mkdir -p ./checkpoint/${s}${t}/${task}
fi

cp $0 model/${task}

echo ">> training ${task}"

nohup fairseq-train ${databin} \
  --save-dir ./checkpoint/${s}${t}/${task} \
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
  --fp16-scale-tolerance 0.1 >./checkpoint/${s}${t}/${task}/train.log 2>&1 &
wait


# for small dataset e.g. en-ro, we just change following settings:
#   --attention-dropout 0.3 \
#   --activation-dropout 0.3 \
#   --dropout 0.3 \
#   --share-all-embeddings \
#   --warmup-updates 4000--lr-period-updates 21000 \
#   --max-update 25000 \
#   --weight-decay 0.0001