# Evaluation Only




```bash
# 1x2080ti machine
python src/training/main.py \
    --linear-frequency 0  --zeroshot-frequency 1 --retrieval-frequency 0  --nlp-eval-frequency 0 --eval-data-dir '/data/Datasets' \
    --retrieval-data '/data/Datasets/RSICD/csv/rsicd_test.csv' --retrieval-images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --retrieval-csv-separator '\t' --retrieval-csv-img-key 'filename' --retrieval-csv-caption-key 'title' \
    --image-model 'ViT-L-14' --image-model-builder 'openclip' \
    --text-model 'IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese' --text-model-builder 'huggingface' --text-pooler 'logits' \
    --pretrained-image-model --pretrained-text-model \
    --logs 'logs/eval'  --name 'Taiyi-CLIP-ViT-L-14-zeroshot'
```


```bash
# 1x2080ti machine
python src/training/main.py \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 1  --nlp-eval-frequency 0 --eval-data-dir '/data/Datasets' \
    --retrieval-data '/data/Datasets/RSICD/csv/rsicd_test.csv' --retrieval-images-dir '/data/Datasets/RSICD/RSICD_images/RSICD_images' \
    --retrieval-csv-separator '\t' --retrieval-csv-img-key 'filename' --retrieval-csv-caption-key 'title' \
    --image-model 'ViT-B-16' --image-model-builder 'openclip' \
    --text-model 'ViT-B-16' --text-model-builder 'openclip' \
    --pretrained-image-model --pretrained-text-model \
    --logs 'logs/eval'  --name 'openclip-ViT-B-16-retrieval'
```





## Downstream Evaluation
- Image Classification (ELEVATER?)
- Image-text Retrieval
- Sentence Similarity
- MS MARCO Passage Retrval...


Then you can perform EVEVATOR evaluations of the model trained by this codebase, by making necessary modifications and run the following commands:

```bash
conda activate vlkd
cd /data/codes/ProtoRKD 
export PYTHONPATH="$PWD/src/training/evaluations:$PWD/src"

# zero-shot:       model_cfg='clip_zeroshot_eval'      mode='zeroshot'\
# few-shot:        model_cfg='cls_linear_or_ft_eval'   mode='linear_probe' num_shots=5 \
# linear prob:     model_cfg='cls_linear_or_ft_eval'   mode='linear_probe' num_shots=-1 \
# fine-tune:       model_cfg='cls_linear_or_ft_eval'   mode='finetune'     num_shots=-1 \

for dataset (caltech101 cifar10 cifar100 country211 dtd eurosat-clip fer2013 fgvc-aircraft-2013b flower102 food101 gtsrb hateful-memes kitti-distance mnist oxford-iiit-pets patchcamelyon rendered-sst2 resisc45-clip stanfordcar voc2007classification)
{       
    #---> REPLACE THIS LINE WITH ONE OF FOUR OPTIONS ABOVE <---#
    log_dir=# <YOUR EXPERIMENT DIR> \
    ckpt_epoch=# <WHICH EPOCH> \
    dataset_root=# <YOUR DATASET DIR> \
    dataset=$dataset \
    disable_hyperparameter_tuning=True \
        bash run_evevater_eval.sh
}
```

for example,
```bash
conda activate vlkd
cd /data/codes/ProtoRKD 
export PYTHONPATH="$PWD/src/training/evaluations:$PWD/src"

for dataset (caltech101 cifar10 cifar100 country211 dtd eurosat-clip fer2013 fgvc-aircraft-2013b flower102 food101 gtsrb hateful-memes kitti-distance mnist oxford-iiit-pets patchcamelyon rendered-sst2 resisc45-clip stanfordcar voc2007classification)
{       
    model_cfg='cls_linear_or_ft_eval'   mode='finetune'     num_shots=-1 \
    log_dir='/data/codes/ProtoRKD/logs/codebase_test/U[mobilenet_v3_large-h2]-L[CLIP-from-RN50]-bs1024-YFCC-56ep-lr1e-5' \
    ckpt_epoch=56 \
    dataset=$dataset \
    disable_hyperparameter_tuning=True \
    dataset_root='/data/codes/ProtoRKD/src/training/evaluations/vision_benchmark/outputs/datasets'\
        bash run_evevater_eval.sh
}

```

Then you can generate submission file for [EvalAI](https://eval.ai/web/challenges/challenge-page/1832/overview). For more details, please see [official instructions](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC#submit-to-leaderboard).


```bash
python src/training/evaluations/vision_benchmark/commands/prepare_submit.py \
  --combine_path 'logs/codebase_test/L[mobilenet_v3_small-h2]-L[CLIP-from-RN50]-bs1024-YFCC-8ep/clip_zeroshot_eval/log/predictions/zeroshot_eval_wiki_False_wnh_False_wnd_False_gpt3_Falseagg_WIKI_AND_GPT3_gpt3count_0'
```

We provide a simple script to summarize the results:
```bash
python src/utils/summarize_ELEVATER_results.py
Input your log dir (end with "../ELEVATER_evaluation/<eval_mode>"):
>>> logs/U[mobilenet_v3_large-h2]-L[CLIP-from-RN50]-bs1024-YFCC-56ep-lr1e-5/ELEVATER_evaluation/zeroshot
                           Dsataset  zeroshot-accuracy%
0                       caltech-101             70.4490
1                          cifar-10             72.8000
2                         cifar-100             37.1700
3                        country211              7.0570
4                               dtd             31.5430
5                      eurosat_clip             25.3000
6                          fer-2013             21.8170
7   fgvc-aircraft-2013b-variants102              5.1620
8                 oxford-flower-102             45.4590
9                          food-101             40.3290
10                            gtsrb              8.8600
11                    hateful-memes             52.4110
12                   kitti-distance             14.3460
13                            mnist             11.0400
14                 oxford-iiit-pets             65.2600
15                   patch-camelyon             50.7600
16                    rendered-sst2             47.8860
17                    resisc45_clip             23.2740
18                    stanford-cars              5.0990
19          voc-2007-classification             77.5720
20                          Average             35.6797
saved to logs/U[mobilenet_v3_large-h2]-L[CLIP-from-RN50]-bs1024-YFCC-56ep-lr1e-5/ELEVATER_evaluation/zeroshot/summary.csv
```

