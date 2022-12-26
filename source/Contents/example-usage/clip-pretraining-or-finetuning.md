# CLIP Pretraining or Finetuning

First, assume that you have already created an environment with [required dependencies](../getting-started/install-dependencies.md), prepared data for [pre-training](../getting-started/prepare-pretraining-data.md) and [downstream evaluations](../getting-started/prepare-downstream-data.md).

Then you can activate the environment and modify the `PYTHONPATH` variable, such that modules can be imported successfully.
```bash
conda activate ITRA
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```

## Contrastive Language Image Pretraining From Scratch

Training a CLIP from scratch is the most straight forward usage of `ITRA`. By specifying `--loss 'InfoNCE'`, the model will contrast image and text samples within a batch.

```bash
# Example command for a 8x2080ti machine
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 14000000 --train-data 'cache/yfcc_nori.csv' --nori-dataset\
    --epochs 8 --save-frequency 8 --batch-size 64 --workers 8 \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/CLIP'
```

## Fine-tune a CLIP 


```bash
# 8x2080ti machine, coco
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 1  --nlp-eval-frequency 0  --eval-data-dir '/data/Datasets' \
    --epochs 10 --save-frequency 10 --batch-size 64 --workers 0 \
    --lr 1e-5 --warmup 100 --wd 0.5 --max-grad-norm 5 \
    --image-model 'ViT-B-32' --image-model-builder 'openclip' --text-model 'ViT-B-32' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO'  --name 'ViT-B-32-finetune'
```

```bash
# 8x2080ti machine, RSICD
torchrun --nproc_per_node 8 -m training.main \
    --train-data '/data/Datasets/RSICD/csv/rsicd_train.csv' --images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --csv-separator '\t' --csv-img-key 'filename' --csv-caption-key 'title' \
    --retrieval-data '/data/Datasets/RSICD/csv/rsicd_test.csv' --retrieval-images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --retrieval-csv-separator '\t' --retrieval-csv-img-key 'filename' --retrieval-csv-caption-key 'title' \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 1  --nlp-eval-frequency 0  --eval-data-dir '/data/Datasets' \
    --epochs 10 --save-frequency 10 --batch-size 64 --workers 4 \
    --lr 5e-5 --warmup 100 --wd 0.5 --max-grad-norm 5 \
    --image-model 'ViT-B-32' --image-model-builder 'openclip' --text-model 'ViT-B-32' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/RSICD'  --name 'ViT-B-32-finetune-lr=5e-5'
```