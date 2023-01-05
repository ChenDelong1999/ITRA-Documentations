# Evaluate Pretrained Models

## Zero-shot Image-text Retrieval

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

```bash
# 1x2080ti machine
python itra/training/main.py \
    --linear-frequency 0  --zeroshot-frequency 0 --retrieval-frequency 0  --nlp-eval-frequency 1 --datasets-dir '/data/Datasets' \
    --retrieval-data 'mscoco_captions' \
    --image-model 'RN50' --image-model-builder 'openclip'  \
    --text-model 'RN50' --text-model-builder 'openclip'  \
    --pretrained-image-model --pretrained-text-model \
    --logs 'logs/MSCOCO-zeroshot'  --name 'RN50x4-openclip-zeroshot-retrieval
    
    
# [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN50', 'cc12m'), ('RN50-quickgelu', 'openai'), ('RN50-quickgelu', 'yfcc15m'), ('RN50-quickgelu', 'cc12m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'), ('RN101-quickgelu', 'openai'), ('RN101-quickgelu', 'yfcc15m'), ('RN50x4', 'openai'), ('RN50x16', 'openai'), ('RN50x64', 'openai'), ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion400m_e31'), ('ViT-B-32', 'laion400m_e32'), ('ViT-B-32', 'laion2b_e16'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-32-quickgelu', 'openai'), ('ViT-B-32-quickgelu', 'laion400m_e31'), ('ViT-B-32-quickgelu', 'laion400m_e32'), ('ViT-B-16', 'openai'), ('ViT-B-16', 'laion400m_e31'), ('ViT-B-16', 'laion400m_e32'), ('ViT-B-16-plus-240', 'laion400m_e31'), ('ViT-B-16-plus-240', 'laion400m_e32'), ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion400m_e31'), ('ViT-L-14', 'laion400m_e32'), ('ViT-L-14', 'laion2b_s32b_b82k'), ('ViT-L-14-336', 'openai'), ('ViT-H-14', 'laion2b_s32b_b79k'), ('ViT-g-14', 'laion2b_s12b_b42k'), ('roberta-ViT-B-32', 'laion2b_s12b_b32k'), ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'), ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k')]
```


Coming soon...

## Zero-shot Image Classification

Coming soon...

## Linear Probing and KNN CClassification

Coming soon...

## Clustering Evaluation

Coming soon...

## Sentence Embedding Evaluation

STS-Benchmark, SICK...

MS MARCO Passage Retrval...

Word embeddings..

Coming soon...

## ELEVATOR Image Classification Benchmark


You can perform EVEVATOR evaluations of the model trained by this codebase, by making necessary modifications and run the following commands:

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

