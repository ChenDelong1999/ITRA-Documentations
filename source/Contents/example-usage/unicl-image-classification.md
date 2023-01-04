
# Image Classification (UniCL)

UniCL: [Unified Contrastive Learning in Image-Text-Label Space](https://arxiv.org/abs/2204.03610)

## Train an Image Classification Model From scratch

Compare to [MMClassification](https://mmclassification.readthedocs.io/zh_CN/latest/papers/resnet.html)
- [resnet18_cifar.py](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/models/resnet18_cifar.py)
- [cifar10_bs16.py](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/datasets/cifar10_bs16.py)
- [cifar10_bs128.py](https://github.com/open-mmlab/mmclassification/blob/master/configs/_base_/schedules/cifar10_bs128.py)

```bash
# Single GPU classification
python itra/training/main.py \
    --train-data 'CIFAR10' \
    --linear-frequency 20  --zeroshot-frequency 20 --datasets-dir '/data/Datasets' \
    --epochs 200 --save-frequency 0 --batch-size 128 --workers 4 \
    --opt 'sgd' --lr 0.1 --warmup 100 --weight_decay 0.0001 \
    --image-model 'resnet18' --image-model-builder 'torchvision' --image-resolution 32  --image-head-n-layers 1 \
    --pretrained-text-model \
    --text-model 'RN50' --text-model-builder 'openclip' --lock-text-model --text-head-n-layers 1  \
    --loss 'CrossEntropy' --joint-projection-dim 10 \
    --report-to tensorboard --logs 'logs/UniCL-Classification'  --name 'resnet18(scratch)-CIFAR10-200ep-CrossEntropy+linear_eval'
    
# Single GPU classification
python itra/training/main.py \
    --train-data 'CIFAR10' \
    --linear-frequency 5 --zeroshot-frequency 5 --datasets-dir '/data/Datasets' \
    --epochs 200 --save-frequency 0 --batch-size 128 --workers 4 \
    --opt 'sgd' --lr 0.1 --warmup 100 --weight_decay 0.0001 \
    --image-model 'resnet18' --image-model-builder 'torchvision' --image-resolution 32  --image-head-n-layers 1 \
    --pretrained-text-model \
    --text-model 'RN50' --text-model-builder 'openclip' --lock-text-model --text-head-n-layers 1  \
    --loss 'InfoNCE' --joint-projection-dim 1024 \
    --report-to tensorboard --logs 'logs/UniCL-Classification'  --name 'resnet18(scratch)-CIFAR10-200ep-InfoNCE+linear_eval'
```


## Fine-tuning CLIP for ImageNet Classification

Re-implement [this paper](https://arxiv.org/abs/2212.06138).