
# Best Practice 
## Model Architechture

### **Image Backbone**

- **From `OpenCLIP` (v2.0.2)**. [OpenCLIP](https://github.com/mlfoundations/open_clip) is an open source implementation of [OpenAI's CLIP](https://github.com/openai/CLIP) (Contrastive Language-Image Pre-training). To check all supported model architecture and pretrained weigths, run:

    ```python
    import open_clip
    open_clip.list_pretrained()
    ```

    ```bash
    --image-model-builder 'openclip' --image-model 'RN50' \
    --image-model-builder 'openclip' --image-model 'RN50' --image-model-tag 'openclip' --pretrained-image-model \
    --image-model-builder 'openclip' --image-model 'RN50' --image-model-tag 'yfcc15m' --pretrained-image-model  \
    --image-model-builder 'openclip' --image-model 'RN50' --image-model-tag 'cc12m' --pretrained-image-model  \
    ...
    ```

- **From `Torchvision` (v0.12)**. To check all supported model architecture and pretrained weigths, run the following command or see [this page](https://pytorch.org/vision/0.12/models.html).

    ```python
    import torchvision
    torchvision.models.__dict__.keys()
    ```
    
    ```bash
    --image-model-builder 'torchvision' --image-model 'resnet50' \
    --image-model-builder 'torchvision' --image-model 'resnet50' --pretrained-image-model \
    --image-model-builder 'torchvision' --image-model 'alexnet' \
    --image-model-builder 'torchvision' --image-model 'convnext_tiny' \
    --image-model-builder 'torchvision' --image-model 'wide_resnet50_2' \
    --image-model-builder 'torchvision' --image-model 'vgg11' \
    --image-model-builder 'torchvision' --image-model 'squeezenet1_0' \
    --image-model-builder 'torchvision' --image-model 'inception_v3' \
    --image-model-builder 'torchvision' --image-model 'mobilenet_v3_small' \
    --image-model-builder 'torchvision' --image-model 'mnasnet0_5' \
    --image-model-builder 'torchvision' --image-model 'shufflenet_v2_x0_5' \
    --image-model-builder 'torchvision' --image-model 'efficientnet_b0' \
    --image-model-builder 'torchvision' --image-model 'regnet_y_400mf' \
    --image-model-builder 'torchvision' --image-model 'vit_b_16' \
    ```


- **From `Torch Hub`**.
    ```python
    import torch
    for github in ['swav', 'dino', 'vicreg', 'barlowtwins', 'swag', 'deit']:
        print(f'{github}:\t', torch.hub.list(f'facebookresearch/{github}'))
    ```

    ```bash    
    --image-model-builder 'torchhub' --image-model 'resnet50' --image-model-tag 'facebookresearch/swav:main' \
    --image-model-builder 'torchhub' --image-model 'dino_vits16' --image-model-tag 'facebookresearch/dino:main' \
    --image-model-builder 'torchhub' --image-model 'resnet50' --image-model-tag 'facebookresearch/vicreg:main' \
    --image-model-builder 'torchhub' --image-model 'resnet50' --image-model-tag 'facebookresearch/barlowtwins:main' \
    --image-model-builder 'torchhub' --image-model 'regnety_16gf' --image-model-tag 'facebookresearch/swag:main' \
    ...
    ```

    https://github.com/facebookresearch/VICRegL
    import torch
    model = torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p9')
    model = torch.hub.load('facebookresearch/vicregl:main', 'resnet50_alpha0p75')
    model = torch.hub.load('facebookresearch/vicregl:main', 'convnext_small_alpha0p9')
    model = torch.hub.load('facebookresearch/vicregl:main', 'convnext_small_alpha0p75')
    model = torch.hub.load('facebookresearch/vicregl:main', 'convnext_base_alpha0p9')
    model = torch.hub.load('facebookresearch/vicregl:main', 'convnext_base_alpha0p75')
    model = torch.hub.load('facebookresearch/vicregl:main', 'convnext_xlarge_alpha0p75')

    For more details, see:
    - https://github.com/facebookresearch/swav
    - https://github.com/facebookresearch/dino
    - https://github.com/facebookresearch/vicreg
    - https://github.com/facebookresearch/barlowtwins
    - https://github.com/facebookresearch/SWAG
    - https://github.com/facebookresearch/deit/blob/main/README_deit.md



### **Text Backbone**
- **From `OpenCLIP`**. Here the alternatives of the text encoder are exactly the same as OpenCLIP's image backbone.

- **From HuggingFace🤗Transformers**. For more details, see [HuggingFace Transformers](https://huggingface.co/docs/transformers). Currently, only 'from pretrained' mode is supported (i.e., you cannot train a huggingface transformer from scratch now). Standard models like BERT/RoBERTa are supported, but whether other models are also supported is not sure...

- **From Sentence Transformers**. The [Sentence Transformers](https://www.sbert.net) liberary provides powerfull sentence embeddings. Please see [pretrained models](https://www.sbert.net/docs/pretrained_models.html) for more detials. Loading sentence transformers via huggingface and specify `--text-pooler='mean'` is recommended, though it is also supported to load the model via sentence transformer:

    ```bash
    # recommended: 
    --text-model-builder 'huggingface'  --text-model 'sentence-transformers/all-mpnet-base-v2' --text-pooler='mean' 
    # not recommended:
    --text-model-builder 'sbert'  --text-model 'all-mpnet-base-v2' 
    ```

    However, it seems that word embedding models ([GloVe](https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d) and [Komninos](https://huggingface.co/sentence-transformers/average_word_embeddings_komninos)) in sentence-transformers cannot be loaded via huggingface.

- **From Adapter-Transformers**. The [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers) liberary enables Delta-tuning on popular huggingface transformers. See [Model Overview](https://docs.adapterhub.ml/model_overview.html) for available adaptations, and see the [Docs](https://docs.adapterhub.ml/) and [AdapterHub](https://adapterhub.ml/) for more details.

    We have made the following adapters available in this codebase:

    | Method                                                                                                        | args.adapter       |         |
    |---------------------------------------------------------------------------------------------------------------|--------------------|------------|
    | [Bottleneck   adapters](https://docs.adapterhub.ml/overview.html#bottleneck-adapters)                         | `bottleneck_adapter` |          |
    | [Language Adapters](https://docs.adapterhub.ml/overview.html#language-adapters-invertible-adapters)           | `lang_adapter`       |          |
    | [Prefix   Tuning](https://docs.adapterhub.ml/overview.html#prefix-tuning)                                     | `prefix_tuning`      |          |
    | [Compacter](https://docs.adapterhub.ml/overview.html#compacter)                                               | `dummy`              |          |
    | [LoRA](https://docs.adapterhub.ml/overview.html#lora)                                                         | `lora_adapter`       |          |
    | [(IA)^3](https://docs.adapterhub.ml/overview.html#ia-3)                                                       | `ia3_adapter`        |          |
    | [Mix-and-Match   Adapters](https://docs.adapterhub.ml/overview.html#mix-and-match-adapters)                   | `mam_adapter`        |          |
    | [UniPELT](https://docs.adapterhub.ml/overview.html#unipelt)                                                   | `unipelt`            |          |

+-------------------------+---------------------+----------------------------------------------------------------------------------+---+---+---+---+---+---+---+
| Method                  | args.adapter        | Doc                                                                              |   |   |   |   |   |   |   |
+=========================+=====================+==================================================================================+===+===+===+===+===+===+===+
| Bottleneck adapters     | bottleneck_adapter  | https://docs.adapterhub.ml/overview.html#bottleneck-adapters                     |   |   |   |   |   |   |   |
| Language Adapters       | lang_adapter        | https://docs.adapterhub.ml/overview.html#language-adapters-invertible-adapters   |   |   |   |   |   |   |   |
| Prefix Tuning           | prefix_tuning       | https://docs.adapterhub.ml/overview.html#prefix-tuning                           |   |   |   |   |   |   |   |
| Compacter               | dummy               | https://docs.adapterhub.ml/overview.html#compacter                               |   |   |   |   |   |   |   |
| LoRA                    | lora_adapter        | https://docs.adapterhub.ml/overview.html#lora                                    |   |   |   |   |   |   |   |
| (IA)^3                  | ia3_adapter         | https://docs.adapterhub.ml/overview.html#ia-3                                    |   |   |   |   |   |   |   |
| Mix-and-Match Adapters  | mam_adapter         | https://docs.adapterhub.ml/overview.html#mix-and-match-adapters                  |   |   |   |   |   |   |   |
| UniPELT                 | unipelt             | https://docs.adapterhub.ml/overview.html#unipelt                                 |   |   |   |   |   |   |   |
|                         |                     |                                                                                  |   |   |   |   |   |   |   |
+-------------------------+---------------------+----------------------------------------------------------------------------------+---+---+---+---+---+---+---+


### **Projection Head**

- Linear projection head

- [DINO MLP Head](https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/vision_transformer.py#L257) (optionally with a prototype layer in the last)



## Loss Function

| Loss        | Original Task | Paper                                                                                            | Source Implementation                                                          | Uni-Directional | Need Prototype Layer |
|-------------|---------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|-----------------|----------------------|
| InfoNCE     | Alignment     | Learning Transferable Visual Models From Natural Language Supervision                            | https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py#L65 |                 |                      |
| SimReg      | KD            | SimReg:   Regression as a Simple Yet Effective Tool for Self-supervised Knowledge   Distillation | https://github.com/UCDvision/simreg/blob/main/simreg.py#L122                   |                 |                      |
| RKD         | KD            | Relational Knowledge Distillation                                                                | https://github.com/lenscloth/RKD/blob/master/metric/loss.py#L136               |                 |                      |
| CompRess-1q | KD            | CompRess: Self-Supervised Learning by Compressing Representations                                | https://github.com/UMBCvision/CompRess/blob/master/nn/compress_loss.py#L67     | &#10004;        |                      |
| CompRess-2q | KD            | CompRess: Self-Supervised Learning by Compressing Representations                                | https://github.com/UMBCvision/CompRess/blob/master/nn/compress_loss.py#L89     |                 |                      |
| SEED        | KD            | SEED: Self-supervised Distillation For Visual Representation                                     | https://github.com/jacobswan1/SEED/blob/master/tools/utils.py#L188             | &#10004;        |                      |
| VICReg      | SSL           | VICReg: Variance-Invariance-Covariance Regularization For Self-Supervised   Learning             | https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py#L184       |                 |                      |
| BarlowTwins | SSL           | Barlow Twins: Self-Supervised Learning via Redundancy Reduction                                  | https://github.com/facebookresearch/barlowtwins/blob/main/main.py#L187         |                 |                      |
| DINO        | SSL           | Emerging Properties in Self-Supervised Vision Transformers                                       | https://github.com/facebookresearch/dino/blob/main/main_dino.py#L363           | &#10004;        | &#10004;             |




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

