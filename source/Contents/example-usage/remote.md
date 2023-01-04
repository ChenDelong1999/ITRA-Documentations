
## Plus Classification Dataset (掉点..)


```bash
# COCO along
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions_x0.1' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 30 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'UniCL' \
    --report-to tensorboard --logs 'logs/MSCOCOx0.1-RN50'  --name '30ep-bs800-lr3125e-8-wd1.0-naive-baseline-no-warmup'    
    
# ImageNet50k along
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'ImageNet-50k' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 30 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8  --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'UniCL' \
    --report-to tensorboard --logs 'logs/MSCOCOx0.1-RN50'  --name '30ep-bs800-lr3125e-8-wd1.0-imagenet50k-only'    
    
# ImageNet50k + COCO
torchrun --nproc_per_node 8 -m training.main \
    --train-data 'mscoco_captions_x0.1,ImageNet-50k' --retrieval-data 'mscoco_captions' \
    --retrieval-frequency 1 --datasets-dir '/data/Datasets' \
    --epochs 30 --save-frequency 0 --batch-size 100 --workers 2 \
    --lr 3125e-8 --weight_decay 1.0 --max-grad-norm 5 \
    --image-model 'RN50' --image-model-builder 'openclip' --text-model 'RN50' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'UniCL' \
    --report-to tensorboard --logs 'logs/MSCOCOx0.1-RN50'  --name '30ep-bs800-lr3125e-8-wd1.0-imagenet50k+coco'    
```

### RSICD?

```bash
# 8x2080ti machine, RSICD
torchrun --nproc_per_node 8 -m training.main \
    --train-data '/data/Datasets/RSICD/csv/rsicd_train.csv' --images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --csv-separator '\t' --csv-img-key 'filename' --csv-caption-key 'title' \
    --retrieval-data '/data/Datasets/RSICD/csv/rsicd_test.csv' --retrieval-images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --retrieval-csv-separator '\t' --retrieval-csv-img-key 'filename' --retrieval-csv-caption-key 'title' \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 1  --nlp-eval-frequency 0  --datasets-dir '/data/Datasets' \
    --epochs 10 --save-frequency 10 --batch-size 64 --workers 4 \
    --lr 5e-5 --warmup 100 --wd 0.5 --max-grad-norm 5 \
    --image-model 'ViT-B-32' --image-model-builder 'openclip' --text-model 'ViT-B-32' --text-model-builder 'openclip'\
    --pretrained-image-model --pretrained-text-model \
    --loss 'InfoNCE' \
    --report-to tensorboard --logs 'logs/RSICD'  --name 'ViT-B-32-finetune-lr=5e-5'
```
