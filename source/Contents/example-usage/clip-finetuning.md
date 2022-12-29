# Fine-tuning CLIP for MS-COCO Retrieval

[comment]: <> (**TL;DR**: we fine-tuned a CLIP)


## Zero-shot Evaluation

CLIP is a strong model for zero-shot image text retrieval. See [paper-with-code leader board](https://paperswithcode.com/sota/zero-shot-cross-modal-retrieval-on-coco-2014) for performance comparison.

|           Backbone          | # Params all (M) | # Params image (M) | # Params text (M) |   I2T R@1  |  I2T R@5I  |   I2TR@10  |   T2I R@1  |   T2I R@5  |  T2I R@10  | **Mean Recall** |
|:---------------------------:|:----------------:|:------------------:|:-----------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:---------------:|
|             RN50            |          102.01  |             38.32  |            63.69  |     48.06  |     73.88  |     83.02  |     28.31  |     52.96  |     64.10  |      **58.39 ** |
|            RN101            |          119.69  |             56.26  |            63.43  |     49.80  |     74.42  |     82.72  |     30.18  |     54.15  |     65.28  |      **59.43 ** |
|           RN50x16           |          290.98  |            167.33  |           123.65  |     55.38  |     78.24  |     86.30  |     35.24  |     59.47  |     69.58  |      **64.04 ** |
|           ViT-B-32          |          151.28  |             87.85  |            63.43  |     50.02  |     75.00  |     83.24  |     30.36  |     54.77  |     66.09  |      **59.91 ** |
|           ViT-B-16          |          149.62  |             86.19  |            63.43  |     51.72  |     76.76  |     84.26  |     32.70  |     57.77  |     68.26  |      **61.91 ** |
|           ViT-L-14          |          427.94  |            304.29  |           123.65  |     56.08  |     79.60  |     86.90  |     35.33  |     59.96  |     70.15  |      **64.67 ** |
|       **ViT-L-14-336**      |      **427.94 ** |        **304.29 ** |       **123.65 ** | **57.46 ** | **80.34 ** | **87.58 ** | **36.09 ** | **60.66 ** | **70.76 ** |      **65.48 ** |
| **ViT-L-14-336 (official)** |      **427.94 ** |        **304.29 ** |       **123.65 ** |  **58.4 ** |  **81.5 ** |  **88.1 ** |  **37.8 ** |  **62.4 ** |  **72.2 ** |      **66.73 ** |

For ViT-L-14-336, there is a small gap between our implementation and the officially reported results. We suspect it is caused by image pre-processing: the above re-implementation uses the default `Resize`ransform [as implemented in the official CLIP codes](https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/clip.py#L79), while COCO images are mostly not square, it creates a small train-test domain gap due to distortion. If we alternatively use a `ResizeMaxSize` as implemented [here](https://github.com/mlfoundations/open_clip/blob/3ed21be93e3b9493024dbb42b4461825b5c650a6/src/open_clip/transform.py#L13), the results surpass the official reported performance.

|     Backbone     |     Pre-process    |     I2T R@1 |     I2T R@5I |     I2TR@10 |     T2I R@1 |     T2I R@5 |     T2I R@10 |    Mean Recall |
|:----------------:|:------------------:|:-----------:|:------------:|:-----------:|:-----------:|:-----------:|:------------:|:--------------:|
|   ViT-L-14-336   |       Resize       |      57.46  |       80.34  |      87.58  |      36.09  |      60.66  |       70.76  |         65.48  |
|   ViT-L-14-336   | Official (unknown) |       58.4  |        81.5  |   **88.1 ** |       37.8  |       62.4  |        72.2  |         66.73  |
| **ViT-L-14-336** |  **ResizeMaxSize** |  **59.20 ** |   **81.70 ** |      87.96  |  **39.02 ** |  **63.86 ** |   **73.52 ** |     **67.54 ** |

Changing `Resize` into `ResizeMaxSize` brings +2.06 improvement for ViT-L-336. However, we find that the benifit of this modification is not consistent across different backbones. As shown in the following table, generally, `ResizeMaxSize` is more beneficial for large models, and especially the models that have been trained to process HD images (ViT-L-14 v.s. ViT-L-14-336).

|                                 Backbone                                |  RN50 |  RN101 | RN50x16 | ViT-B-32 | ViT-B-16 | ViT-L-14 | ViT-L-14-336 |
|:-----------------------------------------------------------------------:|:-----:|:------:|:-------:|:--------:|:--------:|:--------:|:------------:|
| Performance improvement  (mean recall) by changing to   `ResizeMaxSize` | +0.45  | -0.13  |  +0.10   |  -0.74   |   +0.83   |   +0.96   |     +2.06     |

Therefore, to keep it simple, we will use the default `Resize` transform in the following experiments.

## Getting Started

First, assume that you have already created an environment with [required dependencies](../getting-started/install-dependencies.md), prepared csv datasets for [pre-training](../getting-started/prepare-pretraining-data.md) and [downstream evaluations](../getting-started/prepare-downstream-data.md).
    
Then you can activate the environment and modify the `PYTHONPATH` variable, such that modules can be imported successfully.
```bash
conda activate ITRA
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```

[Paper-with-code Leaderboard](https://paperswithcode.com/sota/cross-modal-retrieval-on-coco-2014)

Our baseline setting are listed as follows:

- backbone: ResNet50
- batch_size: 32x8=256
- dataset_size: 118287
- epochs: 10
- lr: 1e-05
- opt: adamw
- use_bn_sync: False
- warmup: 100
- weight_decay: 0.5
<details>
<summary>Training Command</summary>

```bash
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --eval-data-dir '/data/Datasets' \
    --epochs 10 --save-frequency 0 --batch-size 32 --workers 2 \
    --lr 1e-5 --warmup 100 --weight_decay 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50'  --name '10ep-bs256-lr1e-5-wd0.5'
```
</details>

This configuration significantly improves the retrieval performance (58.39→73.98, +15.59).

|  Backbone  |   I2T R@1 |    I2T R@5I |    I2TR@10 |    T2I R@1 |    T2I R@5 |    T2I R@10 | Mean Recall |
|:----------:|:---------:|:-----------:|:----------:|:----------:|:----------:|:-----------:|:-----------:|
|  Zero-shot |    48.06  |      73.88  |     83.02  |     28.31  |     52.96  |      64.10  |      58.39  |
| Fine-tuned |    64.84  |      86.62  |     92.30  |     44.99  |     72.76  |      82.34  |      73.98  |





## Hyper-parameter Tuning



## Finetuning with tricks

Related works
- [CLIP Itself is a Strong Fine-tuner: Achieving 85.7% and 88.0% Top-1 Accuracy with ViT-B and ViT-L on ImageNet](https://arxiv.org/abs/2212.06138)
- [Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903)

- Partial finetuning
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
```


| Finetune mode | Backbone | Trainable Parameters (M) | GPU Memory (MB) |
|---------------|------|------|------|
| Naive finetune (all parameters) | ResNet-50 | 102.01 | 5836 |


- EMA
```bash
--model_ema --model_ema_decay 0.998 \
```


- Layer-wise Learning Rate Decay (LLDR)

```bash
--layer_decay_image 0.9 --layer_decay_text 1 \
```

- Wise-FT
Evaluate the model with weight space ensemble [Wise-FT](https://arxiv.org/abs/2109.01903)
```bash
--eval-with-wise-ft 0.5 \
```


## Fine-tuning with Classification Dataset (UniCL)

...

```bash
# Vanilla Naive finetuning
# 8x2080ti machine, ms coco 10 epoch
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --eval-data-dir '/data/Datasets' \
    --epochs 10 --save-frequency 0 --batch-size 32 --workers 2 \
    --lr 1e-5 --warmup 100 --weight_decay 1.25 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50'  --name '10ep-bs256-lr1e-5-wd1.25'

    
# Vanilla Naive finetuning
# 8x2080ti machine, ms coco 10 epoch
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --eval-data-dir '/data/Datasets' \
    --epochs 10 --save-frequency 0 --batch-size 32 --workers 2 \
    --lr 1e-5 --warmup 100 --weight_decay 1.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50'  --name '10ep-bs256-lr1e-5-wd1.5'
    
    
    
# Vanilla Naive finetuning
# 8x2080ti machine, ms coco 10 epoch
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --eval-data-dir '/data/Datasets' \
    --epochs 30 --save-frequency 0 --batch-size 32 --workers 2 \
    --lr 1e-5 --warmup 100 --weight_decay 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50'  --name '30ep-bs256-lr1e-5-wd0.5'
    
# Vanilla Naive finetuning
# 8x2080ti machine, ms coco 10 epoch
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --eval-data-dir '/data/Datasets' \
    --epochs 30 --save-frequency 0 --batch-size 32 --workers 2 \
    --lr 2e-5 --warmup 100 --weight_decay 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50'  --name '30ep-bs256-lr2e-5-wd0.5'
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



## Single GPU
```bash
# 2080 ti, Baseline
python src/training/main.py \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --eval-data-dir '/data/Datasets' \
    --epochs 10 --save-frequency 0 --batch-size 100 --workers 8 \
    --lr 5e-6 --warmup 100 --weight_decay 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-single-gpu'  --name '10ep-bs100-lr5e-6-wd0.5'
    
    
# 2080 ti, Baseline + tricks
python src/training/main.py \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --eval-data-dir '/data/Datasets' \
    --epochs 10 --save-frequency 0 --batch-size 100 --workers 8 \
    --lr 5e-6 --warmup 100 --weight_decay 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --lock-image-partial 'layer1,layer2' \
    --lock-text-partial 'resblocks.0,resblocks.1,resblocks.2,resblocks.3,resblocks.4,resblocks.5,resblocks.6' \
    --layer_decay_image 0.8 --layer_decay_text 0.8 \
    --model_ema --model_ema_decay 0.998 \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-single-gpu'  --name '10ep-bs100-lr5e-6-wd0.5-lock_image[2blocks]-lock_text[0-6]-LLRD(0.8)-EMA(0.998)'
```
