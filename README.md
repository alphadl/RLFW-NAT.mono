# RLFW-NAT.mono

Code for **"Redistributing Low-Frequency Words: Making the Most of Monolingual Data in Non-Autoregressive Translation"** (ACL 2022).

We use **monolingual knowledge distillation**: train an AT teacher on bilingual data, then distill on **monolingual** source (→KDM) and/or target (←KDM, via backward teacher). NAT is trained on the distilled mono data; combining forward + reverse gives **bidirectional monolingual KD** (←→KDM), which improves low-frequency word translation.

[Paper](https://aclanthology.org/2022.acl-long.172.pdf)

## Setup

```bash
git clone https://github.com/alphadl/RLFW-NAT.mono.git
cd RLFW-NAT.mono
pip install -e fairseq_mask/ -e fairseq_lev/
```

## Data

You need **binarized** data produced by monolingual KD (and optionally standard KD). Pipeline:

1. Train an AT teacher on the bilingual parallel data.
2. **Forward mono KD (→KDM):** Use source-side monolingual data; decode with the (forward) AT teacher to get synthetic targets; this gives (mono_src, pseudo_tgt).
3. **Reverse mono KD (←KDM):** Train a backward AT teacher (tgt→src); use target-side monolingual data and decode to get (pseudo_src, mono_tgt).
4. Optionally concatenate →KDM and ←KDM for **bidirectional mono KD** (←→KDM), and/or mix with standard KD data (→KDB).
5. Run fairseq preprocess on the resulting parallel data to get a **databin** directory.

Put the databin path in `DATA` when training/evaluating (see below).

## Training

**Mask-Predict** (CMLM):

```bash
SRC=en TGT=de DATA=/path/to/databin SAVE_DIR=./checkpoint/ende/mask_mono bash train_mask.sh
```

**Levenshtein**:

```bash
SRC=en TGT=de DATA=/path/to/databin SAVE_DIR=./checkpoint/ende/lev_mono bash train_lev.sh
```

Best checkpoint: `SAVE_DIR/checkpoint_best.pt`. For small data (e.g. En-Ro), see the commented options in the scripts (e.g. higher dropout, fewer updates).

## Eval & Test

**Validation / test decoding:**

```bash
DATA=/path/to/databin CHECKPOINT=./checkpoint/ende/mask_mono SUBSET=valid bash eval_mask.sh
DATA=/path/to/databin CHECKPOINT=./checkpoint/ende/mask_mono SUBSET=test bash eval_mask.sh
```

Hypotheses are written to `CHECKPOINT/gen/<SUBSET>.hyp`. For Levenshtein, use `eval_lev.sh`.

**Test + BLEU:**

```bash
DATA=/path/to/databin CHECKPOINT=./checkpoint/ende/mask_mono REF=/path/to/test.de bash test_mask.sh
```

## Pretrained models

| Dataset | Dict | Model |
|---------|------|-------|
| WMT16 En-Ro | [dict (.zip)](https://drive.google.com/uc?id=1o1h5ZsTJOn4Tyb9l3aigqY7bsVXLtwC5) | [model (.zip)](https://drive.google.com/uc?id=1Y6DmB0XtbZ-bnK_nI9O0qDP0QrdWyvF2) |

## Citation

```bibtex
@inproceedings{ding2022redistributing,
  title = {Redistributing Low-Frequency Words: Making the Most of Monolingual Data in Non-Autoregressive Translation},
  author = {Ding, Liang and Wang, Longyue and Liu, Xuebo and Wong, Derek F. and Tao, Dacheng and Tu, Zhaopeng},
  booktitle = {ACL},
  year = {2022}
}
```

## License

CC-BY-NC 4.0 (applies to code and pre-trained models).
