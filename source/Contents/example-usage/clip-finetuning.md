# CLIP Finetuning

First, assume that you have already created an environment with [required dependencies](../getting-started/install-dependencies.md), prepared data for [pre-training](../getting-started/prepare-pretraining-data.md) and [downstream evaluations](../getting-started/prepare-downstream-data.md).

Then you can activate the environment and modify the `PYTHONPATH` variable, such that modules can be imported successfully.
```bash
conda activate ITRA
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```

## Related Work

- [CLIP Itself is a Strong Fine-tuner: Achieving 85.7% and 88.0% Top-1 Accuracy with ViT-B and ViT-L on ImageNet](https://arxiv.org/abs/2212.06138)
- [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903)


## Fine-tune a CLIP for MS-COCO Retrieval

[Paper-with-code Leaderboard](https://paperswithcode.com/sota/cross-modal-retrieval-on-coco-2014)


Finetuning tricks:
```bash
# lock image tower, i.e., Locked Image Tuning (LiT) https://arxiv.org/abs/2111.07991
--lock-image-model \

# lock all weight in image tower, while only train the text tower
--lock-image-partial 'weight' \

# only unlock all weight in image tower, while other params are locked
--lock-image-partial '!weight' --lock-image-model \

# Only train the first layer (transformer block) of the image backbone
--lock-image-partial '!resblocks.0'  --lock-image-model \

# Only unfreeze all bias and norm params, i.e., Bias and Normalization Optimization (BiNor) https://arxiv.org/abs/2203.07190
--lock-image-partial '!bias,!ln,!bn' --lock-text-partial '!bias,!ln' --lock-image-model  --lock-text-model \

# Layer wise learning rate decay
--layer_decay_image 0.9 --layer_decay_text 1 \

# EMA Evaluation
--model_ema --model_ema_decay 0.998 \

# Evaluate the model with weight space ensemble Wise-FT (https://arxiv.org/abs/2109.01903)
--eval-with-wise-ft 0.5 \
```


# Experiment Reports

Learning Rate

| Learning Rate | 1e-5 | 2e-5 | 3e-5 | 5e-5 | 1e-4 | 5e-4 | 1e-3 |
|---------------|------|------|------|------|------|------|------|
| ResNet-50     |      |      |      |      |      |      |      |
| ViT-B-32      |      |      |      |      |      |      |      |


| Finetune mode | Backbone | Trainable Parameters (M) | GPU Memory (MB) |
|---------------|------|------|------|
| Naive finetune (all parameters) | ResNet-50 | 102.01 | 5836 |

# Commands

```bash
# Vanilla Naive finetuning
# 8x2080ti machine, ms coco 10 epoch
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --eval-data-dir '/data/Datasets' \
    --epochs 20 --save-frequency 20 --batch-size 32 --workers 2 \
    --lr 2e-5 --warmup 100 --weight_decay 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50'  --name '20ep-bs256-lr2e-5-wd0.5'
```

```bash
# 8x2080ti machine, ms coco 10 epoch, test
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --eval-data-dir '/data/Datasets' \
    --epochs 10 --save-frequency 20 --batch-size 32 --workers 2 \
    --lr 2e-5 --warmup 100 --weight_decay 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' --model_ema --model_ema_decay 0.98 \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50'  --name '10ep-bs256-lr2e-5-wd0.5-EMA(0.98)'
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