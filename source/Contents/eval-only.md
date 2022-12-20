# Evaluation Only




```bash
# 1x2080ti machine
python src/training/main.py \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 0  --nlp-eval-frequency 0 --eval-data-dir '/data/Datasets' \
    --retrieval-data '/data/Datasets/RSICD/csv/rsicd_test.csv' --retrieval-images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --retrieval-csv-separator '\t' --retrieval-csv-img-key 'filename' --retrieval-csv-caption-key 'title' \
    --image-model 'ViT-H-14' --image-model-builder 'chineseclip' \
    --text-model 'ViT-H-14' --text-model-builder 'chineseclip' \
    --pretrained-image-model --pretrained-text-model \
    --logs 'logs/eval'  --name 'ChineseCLIP-ViT-H-14-zeroshot'
```


```bash
# 1x2080ti machine
python src/training/main.py \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 0  --nlp-eval-frequency 0 --eval-data-dir '/data/Datasets' \
    --retrieval-data '/data/Datasets/RSICD/csv/rsicd_test.csv' --retrieval-images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --retrieval-csv-separator '\t' --retrieval-csv-img-key 'filename' --retrieval-csv-caption-key 'title' \
    --image-model 'xlm-roberta-large-ViT-H-14' --image-model-builder 'openclip' --image-model-tag 'frozen_laion5b_s13b_b90k' \
    --text-model 'xlm-roberta-large-ViT-H-14' --text-model-builder 'openclip' --text-model-tag 'frozen_laion5b_s13b_b90k' \
    --pretrained-image-model --pretrained-text-model \
    --logs 'logs/eval'  --name 'openclip-xlm-roberta-ViT-H-14-zeroshot'
```