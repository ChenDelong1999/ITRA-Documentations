# CLIP (pretraning or re-pretraining)

```bash
conda activate ITRA
```

## CLIP From Scratch

```bash
# 8x2080ti machine
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 4000000 --train-data 'cache/yfcc_nori.csv' --nori-dataset\
    --epochs 28 --save-frequency 28 --batch-size 64 --workers 8 \
    --linear-frequency 1  --zeroshot-frequency 1 --retrieval-frequency 1  --nlp-eval-frequency 1  --eval-data-dir '/data/Datasets' \
    --lr 1e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --distiller 'InfoNCE' \
    --report-to tensorboard --logs 'logs/CLIP'  --name 'from_sratch-bs512-yfcc-8ep'
```

## Finetune a CLIP 