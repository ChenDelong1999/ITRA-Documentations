# Fine-tuning CLIP for MS-COCO Retrieval

In this section, we present an example usage and some empirical guides of fine-tuning CLIP for image-text retrieval. We aim to improve the retrieval performance based on the strong zero-shot retrieval ability (see our [evaluation report](../example-usage/eval-only.md)) of CLIP by fine-tuning CLIP on [MS COCO Captions](https://paperswithcode.com/dataset/coco) training set (118k images) with the InfoNCE loss. Contents and key findings of this section are listed as follows:
- Fine-tuning CLIP on MS COCO training set improves the retrieval mean recall by +15% compared to raw zero-shot retrieval.
- Proper hyper-parameters can bring at least +1% improvement.
- Scale up batch size by partially freeze CLIP weights brings +1% improvement.
- Compared to the zero-shot retrieval mean recall=58.39% of RN50 CLIP, at last we achieve 76.02% mean recall (17.63% improvement) by fine-tuning it with a 8x2080ti machine.

## Getting Started: Naive Fine-tuning Baseline

First, assume that you have already created an environment with [required dependencies](../getting-started/install-dependencies.md), prepared csv datasets for [pre-training](../getting-started/prepare-pretraining-data.md) and [downstream evaluations](../getting-started/prepare-downstream-data.md). Then you can activate the environment and modify the `PYTHONPATH` variable, such that modules can be imported successfully.

```bash
conda activate ITRA
cd path/to/ITRA/
export PYTHONPATH="$PYTHONPATH:$PWD/itra"
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
</details><br>

Under this configuration, fine-tuning significantly improves the retrieval performance (58.39→73.98, +15.59).

|Type|Model|# Params (M)| I2T R@1| I2T R@5I| I2T R@10| T2I R@1| T2I R@5| T2I R@10|Mean Recall|
|---|---|---|---|---|---|---|---|---|---|
|Two-stream|Zero-shot CLIP RN50|102.01|48.06|73.88|83.02|28.31|52.96|64.1|58.39|
|Two-stream|👉 **Fine-tuned CLIP RN50**|102.01|64.84|86.62|92.3|44.99|72.76|82.34|73.98|
|Two-stream|FLIP (ViT-L-14) |427.94|78.9|94.4|97.4|61.2|84.3|90.6|84.5|
|Two-stream|Florence (CoSwin-H)  |637|81.8|95.2||63.2|85.7|||
|Single-stream| BLIP (large) |220|80.6|95.2|97.6|63.1|85.3|91.1|85.5|
|Single-stream|PTP-BLIP (large) |220|84.2|79.3|98.8|68.8|89.5|94.2|88.8|

```eval_rst
.. note ::
  👆 Here `Florence <https://paperswithcode.com/paper/florence-a-new-foundation-model-for-computer>`_ and `PTP-BLIP <https://paperswithcode.com/paper/position-guided-text-prompt-for-vision>`_ are respectively the two-stream and single-stream SoTA retrieval methods at `paper-with-code leaderboard <https://paperswithcode.com/sota/cross-modal-retrieval-on-coco-2014>`_ by 2022.12.
```


---

## Tuning Hyper-parameters

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


**5. ✨ Improved Naive Baseline with Better Hyper-parameters**. Combining all the above hyper-parameter sweep observations together, we increase the mean recall of naive fine-tuning baseline from 73.98 to 75.04.


|               | Baseline Hyper-parameters | ✨ Improved Hyper-parameters |
|---------------|---------------------------|---------------------------|
| backbone:     | ResNet50                  | ResNet50                  |
| batch_size:   | 32x8=256                  | 100x8=800                 |
| epochs:       | 10                        | 15                        |
| lr:           | 1e-05                     | 3.125e-05                 |
| weight_decay: | 0.5                       | 1.0                       |


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
</details><br>

Results:

|Model             | I2T R@1| I2T R@5I| I2T R@10| T2I R@1| T2I R@5| T2I R@10|Mean Recall|
|------------------|--------|--------|--------|--------|--------|--------|----------|
| Baseline          | 64.84  | 86.62  | 92.30  | 44.99  | 72.76  | 82.34  | 73.98    |
| Improved Baseline | 65.34  | 87.44  | 92.84  | 46.70  | 74.45  | 83.47  | 75.04    |

---

## Scaling up Batch Size by Partially Freeze Weights


| Fine-tuning   Streategy                   | Image  Params | Text  Params | Total Trainable Params (M) | %      | I2T R@1 | I2T R@5I | I2TR@10 | T2I R@1 | T2I R@5 | T2I R@10 | Mean Recall |
|-------------------------------------------|---------------|--------------|----------------------------|--------|---------|----------|---------|---------|---------|----------|-------------|
| zero-shot evaluation                      | -             | -            | 0                          | 0.0%   | 48.06   | 73.88    | 83.02   | 28.31   | 52.96   | 64.10    | 58.39       |
| lock CLIP and add linear projection heads | linear head   | linear head  | 2.1                        | 2.1%   | 47.24   | 75.06    | 84.82   | 32.91   | 61.21   | 72.83    | 62.34       |
| lock CLIP and add MLP projection heads    | MLP head      | MLP head     | 16.79                      | 16.5%  | 53.12   | 79.86    | 87.76   | 37.46   | 65.63   | 76.41    | 66.71       |
| lock image tune text                      | -             | All          | 63.69                      | 62.4%  | 62.12   | 85.12    | 91.46   | 42.52   | 70.34   | 80.31    | 71.98       |
| lock text tune image                      | All           | -            | 38.32                      | 37.6%  | 59.78   | 84.10    | 90.86   | 43.57   | 71.02   | 80.76    | 71.68       |
| naïve fine-tuning (improved   baseline)   | All           | All          | 102.01                     | 100.0% | 65.34   | 87.44    | 92.84   | 46.70   | 74.45   | 83.47    | 75.04       |

- lock image and partially fine-tune text

| Text  Params               | projection+ln_final | 11     | 10,11  | 8~11   | 6~11   | 4~11   | 2~11   | 0~11   | All      |
|----------------------------|---------------------|--------|--------|--------|--------|--------|--------|--------|----------|
| Total Trainable Params (M) | 0.53                | 3.68   | 6.83   | 13.13  | 19.44  | 25.74  | 32.05  | 38.35  | 63.69    |
| %                          | 0.5%                | 3.6%   | 6.7%   | 12.9%  | 19.1%  | 25.2%  | 31.4%  | 37.6%  | 62.4%    |
| Mean Recall                | 67.23               | 69.15  | 70.27  | 71.36  | 71.79  | 71.96  | 72.00  | 72.18  | 71.98    |

- lock text and partially fine-tune image

| Image  Params              | attnpool | attnpool,layer4 | attnpool,layer4,3 | attnpool,layer4,3,2 | attnpool,layer4,3,2,1 | All      |
|----------------------------|----------|-----------------|-------------------|---------------------|-----------------------|----------|
| Text  Params               | -        | -               | -                 | -                   | -                     | -        |
| Total Trainable Params (M) | 14.79    | 29.75           | 36.85             | 38.07               | 38.29                 | 38.32    |
| %                          | 14.5%    | 29.2%           | 36.1%             | 37.3%               | 37.5%                 | 37.6%    |
| Mean Recall                | 71.33    | 72.49           | 72.23             | 71.89               | 71.82                 | 71.68    |

- Scale up batchsize



| Fine-tuning   Streategy                                   | Image  Params   | Text  Params | Total Trainable Params (M) | %      | I2T R@1 | I2T R@5I | I2TR@10 | T2I R@1 | T2I R@5 | T2I R@10 | Mean Recall |
|-----------------------------------------------------------|-----------------|--------------|----------------------------|--------|---------|----------|---------|---------|---------|----------|-------------|
| naïve fine-tuning (improved   baseline) bs-800-lr3.125e-5 | All             | All          | 102.01                     | 100.0% | 65.34   | 87.44    | 92.84   | 46.70   | 74.45   | 83.47    | 75.04       |
| bs800-lr3.125e-5                                          | attnpool,layer4 | 0~11         | 68.11                      | 66.8%  | 66.10   | 87.60    | 93.56   | 47.61   | 75.17   | 84.18    | 75.70       |
| bs1792-lr7e-5                                             | attnpool,layer4 | 0~11         | 68.11                      | 66.8%  | 65.95   | 88.30    | 93.66   | 48.08   | 75.71   | 84.42    | 76.02       |


<details>
<summary>Training Command</summary>

```bash    
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 15 --batch-size 224 --workers 4 \
    --lr 7e-5 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model \
    --lock-text-partial 'positional_embedding,token_embedding' \
    --lock-image-partial '!attnpool,!layer4' \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-partial'  --name 'save-lock-image(!attnpool,!layer4)-lock-text(positional_embedding,token_embedding)-bs1792-lr7e-5'
```
</details>
<br>


## More Tricks for Fine-tuning

### Layer-wise Learning Rate Decay (LLDR)

```bash
--layer_decay_image 0.9 --layer_decay_text 1 \
```

```bash
for layer_decay_text in 1.0 0.95 0.9 0.85 0.8 0.75 0.7 0.65 0.6;
do
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 224 --workers 2 \
    --lr 7e-5 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model \
    --lock-text-partial 'positional_embedding,token_embedding' \
    --lock-image-partial '!attnpool,!layer4' \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-LLDR'  --name 'layer_decay_text='$layer_decay_text \
    --layer_decay_text $layer_decay_text; 
done
```

### Exponential Moving Average (EMA)
```bash
--model_ema --model_ema_decay 0.998 \
```


```bash
for model_ema_decay in 0.99999 0.9999 0.9995 0.999 0.995 0.99 0.95 0.9 0.8;
do
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 224 --workers 2 \
    --lr 7e-5 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model \
    --lock-text-partial 'positional_embedding,token_embedding' \
    --lock-image-partial '!attnpool,!layer4' \
    --loss 'InfoNCE' \
    --model_ema --model_ema_decay $model_ema_decay \
    --report-to tensorboard --logs 'logs/MSCOCO-RN50-EMA'  --name 'model_ema_decay='$model_ema_decay;
done
```


### Wise-FT. Evaluate the model with weight space ensemble 

[Wise-FT](https://arxiv.org/abs/2109.01903)

```bash
--eval-with-wise-ft 0.5 \
```


```bash
for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ;
do
python itra/training/main.py \
    --zeroshot-frequency 1 --retrieval-frequency 1 --retrieval-data 'mscoco_captions' --datasets-dir '/data/Datasets' \
    --image-model 'RN50' --image-model-builder 'openclip'  \
    --text-model 'RN50' --text-model-builder 'openclip'  \
    --pretrained-image-model --pretrained-text-model \
    --resume 'logs/MSCOCO-RN50-partial/save-lock-image(!attnpool,!layer4)-lock-text(positional_embedding,token_embedding)-bs1792-lr7e-5/checkpoints/epoch_15.pt' \
    --eval-with-wise-ft $alpha \
    --logs 'logs/MSCOCO-RN50-WiseFT'  --name 'zs+retrieval-WiseFT='$alpha;
done
```


### rsicd retrieval
```bash
# 1x2080ti machine RSICD 对点
torchrun --nproc_per_node 8 -m training.main \
    --train-data '/data/Datasets/RSICD/csv/rsicd_train.csv' --images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --csv-separator '\t' --csv-img-key 'filename' --csv-caption-key 'title' \
    --retrieval-data '/data/Datasets/RSICD/csv/rsicd_test.csv' --retrieval-images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --retrieval-csv-separator '\t' --retrieval-csv-img-key 'filename' --retrieval-csv-caption-key 'title' \
    --retrieval-frequency 1  --datasets-dir '/data/Datasets' \
    --epochs 30 --save-frequency 0 --batch-size 16 --workers 2 \
    --lr 1e-6 --warmup 100 --weight_decay 0.5 --max-grad-norm 5 \
    --image-model 'ViT-L-14-336' --image-model-builder 'openclip' \
    --text-model 'ViT-L-14-336' --text-model-builder 'openclip' \
    --pretrained-image-model --pretrained-text-model \
    --lock-image-model --lock-text-model \
    --lock-image-partial '!ln_post,!resblocks.23,!resblocks.22,!resblocks.21,!resblocks.20,!resblocks.19,!resblocks.18' \
    --lock-text-partial '!text_projection,!ln_final,!resblocks.11,!resblocks.10,!resblocks.9' \
    --loss 'InfoNCE' --layer_decay_image 0.9 --layer_decay_text 0.9 \
    --report-to tensorboard --logs 'logs/RSICD-ViT-L-14'  --name '30ep-b128-lr1e-5-unlock-image-text-last0.75-lldr0.9'
```


python itra/training/main.py \
    --config-yaml 'logs/params.yml' --name 'custom-name'

python itra/training/main.py \
    --episode-size 10000 \
    --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 15 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 1e-4 --warmup 100 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model --lock-image-model --lock-text-model \
    --loss 'InfoNCE' --prompt --n-prompt 4 \
    --report-to tensorboard --logs 'logs/test'  --name 'coco-finetune-nprompt-4'