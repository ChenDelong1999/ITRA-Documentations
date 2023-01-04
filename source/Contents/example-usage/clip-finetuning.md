# Fine-tuning CLIP for MS-COCO Retrieval

[comment]: <> (**TL;DR**: we fine-tuned a CLIP)


## Zero-shot Retrieval Evaluation

CLIP is a strong model for zero-shot image text retrieval. Since the official paper only reports the performance of the largest CLIP ViT-L-14-336 (standard 32 epoch plus an additional pretraining epoch with 336x336 resolution), here we present our evaluation of other architectures of CLIP. See [paper-with-code leader board](https://paperswithcode.com/sota/zero-shot-cross-modal-retrieval-on-coco-2014) for performance comparison with other zero-shot retrieval methods.

|           Backbone          | # Params all (M) | # Params image (M) | # Params text (M) |   I2T R@1  |  I2T R@5  |   I2T R@10  |   T2I R@1  |   T2I R@5  |  T2I R@10  | **Mean Recall** |
|:---------------------------:|:----------------:|:------------------:|:-----------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:---------------:|
|             RN50            |          102.01  |             38.32  |            63.69  |     48.06  |     73.88  |     83.02  |     28.31  |     52.96  |     64.10  |      **58.39 ** |
|            RN101            |          119.69  |             56.26  |            63.43  |     49.80  |     74.42  |     82.72  |     30.18  |     54.15  |     65.28  |      **59.43 ** |
|           RN50x16           |          290.98  |            167.33  |           123.65  |     55.38  |     78.24  |     86.30  |     35.24  |     59.47  |     69.58  |      **64.04 ** |
|           ViT-B-32          |          151.28  |             87.85  |            63.43  |     50.02  |     75.00  |     83.24  |     30.36  |     54.77  |     66.09  |      **59.91 ** |
|           ViT-B-16          |          149.62  |             86.19  |            63.43  |     51.72  |     76.76  |     84.26  |     32.70  |     57.77  |     68.26  |      **61.91 ** |
|           ViT-L-14          |          427.94  |            304.29  |           123.65  |     56.08  |     79.60  |     86.90  |     35.33  |     59.96  |     70.15  |      **64.67 ** |
|       **ViT-L-14-336**      |      **427.94 ** |        **304.29 ** |       **123.65 ** | **57.46 ** | **80.34 ** | **87.58 ** | **36.09 ** | **60.66 ** | **70.76 ** |      **65.48 ** |
| **ViT-L-14-336 (official)** |      **427.94 ** |        **304.29 ** |       **123.65 ** |  **58.4 ** |  **81.5 ** |  **88.1 ** |  **37.8 ** |  **62.4 ** |  **72.2 ** |      **66.73 ** |

For ViT-L-14-336, there is a small gap between our implemented evaluation and the officially reported results. We suspect it is caused by image pre-processing: the above re-implementations use the default `Resize` transform [as implemented in the official CLIP repo](https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/clip.py#L79), while COCO images are mostly not square, it creates a small train-test domain gap due to distortion. If we alternatively use a `ResizeMaxSize` as implemented [here](https://github.com/mlfoundations/open_clip/blob/3ed21be93e3b9493024dbb42b4461825b5c650a6/src/open_clip/transform.py#L13), the results then surpass the official reported performance.

|     Backbone     |     Pre-process    |     I2T R@1 |     I2T R@5I |     I2TR@10 |     T2I R@1 |     T2I R@5 |     T2I R@10 |    Mean Recall |
|:----------------:|:------------------:|:-----------:|:------------:|:-----------:|:-----------:|:-----------:|:------------:|:--------------:|
|   ViT-L-14-336   |       Resize       |      57.46  |       80.34  |      87.58  |      36.09  |      60.66  |       70.76  |         65.48  |
|   ViT-L-14-336   | Official (unknown) |       58.4  |        81.5  |   **88.1 ** |       37.8  |       62.4  |        72.2  |         66.73  |
| **ViT-L-14-336** |  **ResizeMaxSize** |  **59.20 ** |   **81.70 ** |      87.96  |  **39.02 ** |  **63.86 ** |   **73.52 ** |     **67.54 ** |

Changing `Resize` into `ResizeMaxSize` brings +2.06 improvement for ViT-L-14-336. However, we find that the benifit of this modification is not consistent across different backbones. As shown in the following table, generally, `ResizeMaxSize` is more beneficial for large models, and especially the models that have been trained to process HD images (e.g., it is quite beneficial for ViT-L-14-336 but not that much for ViT-L-14).

|                                 Backbone                                |  RN50 |  RN101 | RN50x16 | ViT-B-32 | ViT-B-16 | ViT-L-14 | ViT-L-14-336 |
|:-----------------------------------------------------------------------:|:-----:|:------:|:-------:|:--------:|:--------:|:--------:|:------------:|
| Mean recall improvement by switching to `ResizeMaxSize` | +0.45  | -0.13  |  +0.10   |  -0.74   |   +0.83   |   +0.96   |     +2.06     |

Therefore, to keep it simple, we will use the default `Resize` transform in the following experiments.


## Getting Started: Naive Fine-tuning Baseline

First, assume that you have already created an environment with [required dependencies](../getting-started/install-dependencies.md), prepared csv datasets for [pre-training](../getting-started/prepare-pretraining-data.md) and [downstream evaluations](../getting-started/prepare-downstream-data.md). Then you can activate the environment and modify the `PYTHONPATH` variable, such that modules can be imported successfully.

```bash
conda activate ITRA
cd path/to/ITRA/
export PYTHONPATH="$PYTHONPATH:$PWD/itra"
ulimit -n 100000 # occasionally the dataloader get stuck due to multiprocessing deadlocks, maybe it can reduce the chance. I'm not totally sure...
```

Then we can start to fine-tune a CLIP on MS-COCO captions 2017 training set (118k images). The results should be compared with the [paper-with-code leaderboard](https://paperswithcode.com/sota/cross-modal-retrieval-on-coco-2014). Our baseline setting are listed as follows, we use a single-node machine with 8 NVIDIA GeForce 2080ti GPUs for training, one training epoch takes about 3.5 minutes.
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
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 10 --save-frequency 0 --batch-size 32 --workers 2 \
    --lr 1e-5 --warmup 100 --weight_decay 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50'  --name '10ep-bs256-lr1e-5-wd0.5'
```
</details>

Under this configuration, fine-tuning significantly improves the retrieval performance (58.39→73.98, +15.59).

|Type|Model|# Params (M)| I2T R@1| I2T R@5I| I2T R@10| T2I R@1| T2I R@5| T2I R@10|Mean Recall|
|---|---|---|---|---|---|---|---|---|---|
|Two-stream|Zero-shot CLIP RN50|102.01|48.06|73.88|83.02|28.31|52.96|64.1|58.39|
|Two-stream|👉 **Fine-tuned CLIP RN50**|102.01|64.84|86.62|92.3|44.99|72.76|82.34|73.98|
|Two-stream|FLIP (ViT-L-14) |427.94|78.9|94.4|97.4|61.2|84.3|90.6|84.5|
|Two-stream|Florence (CoSwin-H)  |637|81.8|95.2||63.2|85.7|||
|Single-stream| BLIP (large) |220|80.6|95.2|97.6|63.1|85.3|91.1|85.5|
|Single-stream|PTP-BLIP (large) |220|84.2|79.3|98.8|68.8|89.5|94.2|88.8|

**Note**: [Florence](https://paperswithcode.com/paper/florence-a-new-foundation-model-for-computer) and [PTP-BLIP](https://paperswithcode.com/paper/position-guided-text-prompt-for-vision) are respectively the two-stream and single-stream SoTA retrieval methods at [paper-with-code leaderboard](https://paperswithcode.com/sota/cross-modal-retrieval-on-coco-2014) by 2022.12.



## Hyper-parameter Tuning

**1. Learning Rate**. We vary the learning rate from 5e-6 to 1e-4, and find that **1e-5 and 2e-5 are good for a batch size of 256**. This results confirms the observations in [this paper](https://arxiv.org/abs/2212.06138), where the authors showed that good ImageNet fine-tuning of CILP ViT-B-16 needs a quite small learning rate (2e-5 and 3e-5 for a batch size of 2048).

| Learning Rate | lr5e-6   | lr1e-5       | lr2e-5       | lr3e-5   | lr5e-5   | lr1e-4   |
|---------------|----------|--------------|--------------|----------|----------|----------|
| Mean Recall   | 72.91    | **73.98   ** | **73.97   ** | 73.32    | 72.46    | 69.34    |

**2. Weight Decay**. The author of [SLIP](https://arxiv.org/abs/2112.12750) paper observed that a larger weight decay (0.5) is beneficial for CLIP. Our experiments also showed that **CLIP can also handle a very large value of weight decay** (i.e., 2.50). Here the training data have 118k samples, and we believe that such property can further benefit CLIP fine-tuning when the data is limited. Our results, as shown in the following table, show that **CLIP is pretty robust to weight decay changes**: when vary the value from 0.01 to 2.50, the performance changes in a range of only +- 0.43.

| Weight   Decay | 2.50     | 2.25     | 2.00     | 1.75     | 1.50     | 1.25     | 1.00     | 0.75     | 0.50     | 0.10     | 0.05     | 0.01     |
|----------------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Mean Recall    | 74.07    | 73.94    | 73.87    | 73.84    | 73.94    | 73.94    | 74.05    | 73.87    | 73.98    | 73.93    | 73.64    | 73.79    |

**3. Training Length**. Similar to the experiments in [FLIP](https://arxiv.org/abs/2212.00794), our experiments showed that **scaling training epochs cannot lead to further performance improvement**. Only 5 or 10 epochs are not sufficient, but 15-20 epochs seems already reached the saturation.

|       Epochs       |   5   |   10  |     15    |     20    |   30  |
|:------------------:|:-----:|:-----:|:---------:|:---------:|:-----:|
| Learning Rate=1e-5 | 72.66 | 73.98 |   74.43   | **74.45** | 73.96 |
| Learning Rate=2e-5 | 72.86 | 73.97 | **74.28** |   74.02   | 74.03 |


**4. Batch Size**. It is well known that batch size has a crucial impact for contrastive learning methods. We confirm this point by varying batch size from 32 to 800 (the maximum allowed batch size for ResNet-50 CLIP on a 8x2080ti machine) while changing learning rate according to liner scaling rule. It shows that **scaling down batch size leads to significant performance drop**:

| BatchSize     | 800       | 512      | 256      | 128      | 64       | 32       |
|---------------|-----------|----------|----------|----------|----------|----------|
| Learning Rate | 3.125E-05 | 2.00E-05 | 1.00E-05 | 5.00E-06 | 2.50E-06 | 1.25E-06 |
| Mean Recall   | 74.89     | 74.85    | 73.98    | 72.14    | 69.24    | 65.04    |


**5. Improved Naive Baseline with Better Hyper-parameters ✨**. Combining all the above hyper-parameter sweep observations together, we increase the mean recall of naive fine-tuning baseline from 73.98 to 75.04.

<details>
<summary>Training Command</summary>

```bash
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50'  --name '15ep-bs800-lr3125e-8-wd1.0'
```
</details>

|               | Baseline Hyper-parameters | Improved Hyper-parameters |
|---------------|---------------------------|---------------------------|
| backbone:     | ResNet50                  | ResNet50                  |
| batch_size:   | 32x8=256                  | 100x8=800                 |
| epochs:       | 10                        | 15                        |
| lr:           | 1e-05                     | 3.125e-05                 |
| weight_decay: | 0.5                       | 1.0                       |


|Model             | I2T R@1| I2T R@5I| I2T R@10| T2I R@1| T2I R@5| T2I R@10|Mean Recall|
|------------------|--------|--------|--------|--------|--------|--------|----------|
| Baseline          | 64.84  | 86.62  | 92.30  | 44.99  | 72.76  | 82.34  | 73.98    |
| Improved Baseline | 65.34  | 87.44  | 92.84  | 46.70  | 74.45  | 83.47  | 75.04    |


## More Tricks for Fine-tuning

**1. Scaling up Batch Size and Avoid Over-fitting: Partial fine-tuning**

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

<details>
<summary>Training Command</summary>

```bash
# text backbone 逐层解锁
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model \
    --lock-text-model  --lock-text-partial '!ln_final,!text_projection'\
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-partial'  --name 'lock-image-lock-text(!ln_final,!text_projection)'
   
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model \
    --lock-text-model --lock-text-partial '!resblocks.11,!ln_final,!text_projection' \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-partial'  --name 'lock-image-lock-text(!11,!ln_final,!text_projection)'
    
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model \
    --lock-text-model --lock-text-partial '!resblocks.11,!resblocks.10,!ln_final,!text_projection' \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-partial'  --name 'lock-image-lock-text(!10-11,!ln_final,!text_projection)'
    
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model \
    --lock-text-model --lock-text-partial '!resblocks.11,!resblocks.10,!resblocks.9,!resblocks.8,!ln_final,!text_projection' \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-partial'  --name 'lock-image-lock-text(!8-11,!ln_final,!text_projection)'
    
    - - -
    
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model --lock-text-model \
    --lock-text-partial '!resblocks.11,!resblocks.10,!resblocks.9,!resblocks.8,!resblocks.7,!resblocks.6,!ln_final,!text_projection' \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-partial'  --name 'lock-image-lock-text(!6-11,!ln_final,!text_projection)'
    
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model --lock-text-model \
    --lock-text-partial '!resblocks.11,!resblocks.10,!resblocks.9,!resblocks.8,!resblocks.7,!resblocks.6,!resblocks.5,!resblocks.4,!ln_final,!text_projection' \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-partial'  --name 'lock-image-lock-text(!4-11,!ln_final,!text_projection)'
    
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model --lock-text-model \
    --lock-text-partial '!resblocks.11,!resblocks.10,!resblocks.9,!resblocks.8,!resblocks.7,!resblocks.6,!resblocks.5,!resblocks.4,,!resblocks.3,!resblocks.2,!ln_final,!text_projection' \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-partial'  --name 'lock-image-lock-text(!2-11,!ln_final,!text_projection)'
    
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model --lock-text-model \
    --lock-text-partial '!resblocks.11,!resblocks.10,!resblocks.9,!resblocks.8,!resblocks.7,!resblocks.6,!resblocks.5,!resblocks.4,!resblocks.3,!resblocks.2,!resblocks.1,!resblocks.0,!ln_final,!text_projection' \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-partial'  --name 'lock-image-lock-text(!0-11,!ln_final,!text_projection)'
   
```
</details>


**2. Layer-wise Learning Rate Decay (LLDR)**

```bash
--layer_decay_image 0.9 --layer_decay_text 1 \
```

**3. Exponential Moving Average (EMA)**
```bash
--model_ema --model_ema_decay 0.998 \
```

**4. Wise-FT**. Evaluate the model with weight space ensemble [Wise-FT](https://arxiv.org/abs/2109.01903)
```bash
--eval-with-wise-ft 0.5 \
```


## Exploring the Maximum Potential of a Single GPU

```bash
# 2080 ti, Baseline
python src/training/main.py \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 10 --save-frequency 0 --batch-size 100 --workers 8 \
    --lr 5e-6 --warmup 100 --weight_decay 0.5 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-single-gpu'  --name '10ep-bs100-lr5e-6-wd0.5'
    
    
# 2080 ti, Baseline + tricks
python src/training/main.py \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
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
