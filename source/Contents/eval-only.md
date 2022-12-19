# Evaluation Only
```bash
# 1x2080ti machine
python src/training/main.py \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 1  --nlp-eval-frequency 0 --eval-data-dir '/data/Datasets' \
    --retrieval-data '/data/Datasets/RSICD/csv/rsicd_test.csv' --retrieval-images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --retrieval-csv-separator '\t' --retrieval-csv-img-key 'filename' --retrieval-csv-caption-key 'title' \
    --image-model 'ViT-B-16' --image-model-builder 'openclip' --text-model 'ViT-B-16' --text-model-builder 'openclip' \
    --pretrained-image-model --pretrained-text-model \
    --logs 'logs/eval'  --name 'CLIP-RN50-retrieval'
```