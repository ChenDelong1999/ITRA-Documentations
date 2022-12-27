# CLIP Finetuning

First, assume that you have already created an environment with [required dependencies](../getting-started/install-dependencies.md), prepared data for [pre-training](../getting-started/prepare-pretraining-data.md) and [downstream evaluations](../getting-started/prepare-downstream-data.md).

Then you can activate the environment and modify the `PYTHONPATH` variable, such that modules can be imported successfully.
```bash
conda activate ITRA
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```

## Related Work

- [CLIP Itself is a Strong Fine-tuner: Achieving 85.7% and 88.0% Top-1 Accuracy with ViT-B and ViT-L on ImageNet](https://arxiv.org/abs/2212.06138)


## Fine-tune a CLIP for MS-COCO Retrieval

[Paper-with-code Leaderboard](https://paperswithcode.com/sota/cross-modal-retrieval-on-coco-2014)

```bash
# 8x2080ti machine, ms coco 3 epoch
torchrun --nproc_per_node 8 -m training.main \
    --episode-size 39429 --train-data 'mscoco_captions' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --eval-data-dir '/data/Datasets' \
    --epochs 9 --save-frequency 9 --batch-size 32 --workers 2 \
    --lr 1e-5 --warmup 100 --wd 0.5 --max-grad-norm 5 \
    --image-model 'ViT-B-16' --image-model-builder 'openclip' --text-model 'ViT-B-16' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' --eval-with-wise-ft 0.5 \
    --report-to tensorboard --logs 'logs/MSCOCO-ViT-B-16'  --name '3ep-bs256-lr1e-5-WiseFT(0.5)'
```

Partial Finetune Examples:
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