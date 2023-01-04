# CLIP Pretraining

First, assume that you have already created an environment with [required dependencies](../getting-started/install-dependencies.md), prepared data for [pre-training](../getting-started/prepare-pretraining-data.md) and [downstream evaluations](../getting-started/prepare-downstream-data.md).

Then you can activate the environment and modify the `PYTHONPATH` variable, such that modules can be imported successfully.
```bash
conda activate ITRA
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```

## Standard Contrastive Language Image Pretraining From Scratch

Training a CLIP from scratch is the most straight forward usage of `ITRA`. By specifying `--loss 'InfoNCE'`, the model will contrast image and text samples within a batch.

```bash
# Example command for a 8x2080ti machine
torchrun --nproc_per_node 8 -m training.main \
    --dataset-size 14000000 --episode-size 14000000 --train-data 'cache/yfcc_nori.csv' --nori-dataset\
    --epochs 8 --save-frequency 8 --batch-size 64 --workers 8 \
    --lr 5e-4 --warmup 2000 --wd 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/example-usage/clip-pretraining/YFCC14M-8_epoch-RN50'
```


## Train a Tiny CLIP

- AlexNet, MobileNet?
- Small SBERT?
- GloVe Embeddings?