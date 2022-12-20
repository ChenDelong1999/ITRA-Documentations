# Load Pretrained Weights

## Pretrained Multi-modal Models
- **From `OpenCLIP` (v2.0.2)**. [OpenCLIP](https://github.com/mlfoundations/open_clip) is an open source implementation of [OpenAI's CLIP](https://github.com/openai/CLIP) (Contrastive Language-Image Pre-training). To check all supported model architecture and pretrained weigths, run:

    ```python
    import open_clip
    open_clip.list_pretrained()
    # [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN50', 'cc12m'), ('RN50-quickgelu', 'openai'), ('RN50-quickgelu', 'yfcc15m'), ('RN50-quickgelu', 'cc12m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'), ('RN101-quickgelu', 'openai'), ('RN101-quickgelu', 'yfcc15m'), ('RN50x4', 'openai'), ('RN50x16', 'openai'), ('RN50x64', 'openai'), ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion400m_e31'), ('ViT-B-32', 'laion400m_e32'), ('ViT-B-32', 'laion2b_e16'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-32-quickgelu', 'openai'), ('ViT-B-32-quickgelu', 'laion400m_e31'), ('ViT-B-32-quickgelu', 'laion400m_e32'), ('ViT-B-16', 'openai'), ('ViT-B-16', 'laion400m_e31'), ('ViT-B-16', 'laion400m_e32'), ('ViT-B-16-plus-240', 'laion400m_e31'), ('ViT-B-16-plus-240', 'laion400m_e32'), ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion400m_e31'), ('ViT-L-14', 'laion400m_e32'), ('ViT-L-14', 'laion2b_s32b_b82k'), ('ViT-L-14-336', 'openai'), ('ViT-H-14', 'laion2b_s32b_b79k'), ('ViT-g-14', 'laion2b_s12b_b42k'), ('roberta-ViT-B-32', 'laion2b_s12b_b32k'), ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'), ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k')]
    ```

    To load the official pretrained CLIP (ResNet-50):
    ```bash
    ... \
    --image-model 'RN50' --image-model-builder 'openclip' \
    --text-model 'RN50' --text-model-builder 'openclip' \
    --pretrained-image-model --pretrained-text-model \
    ```

    Optionally, you can load CLIP models pretrained by OpenCLIP instead of OpenAI by specifying `--image-model-tag` and `--text-model-tag`. For example, to load the [ViT-H-14 pretrained on LAION-2B](https://github.com/mlfoundations/open_clip#vit-h14-224x224):
    ```bash
    ... \
    --image-model 'ViT-H-14' --image-model-builder 'openclip' --image-model-tag 'laion2b_s32b_b79k' \
    --text-model 'ViT-H-14' --text-model-builder 'openclip'  --text-model-tag 'laion2b_s32b_b79k' \
    --pretrained-image-model --pretrained-text-model \
    ```

- **From `ChineseCLIP` (v1.4)**. [ChineseCLIP](https://github.com/OFA-Sys/Chinese-CLIP) is the Chinese version of CLIP. We use a large-scale Chinese image-text pair dataset (~200M) to train the model, and we hope that it can help users to conveniently achieve image representation generation, cross-modal retrieval and zero-shot image classification for Chinese data. This repo is based on OpenCLIP project.
  
  The ChineseCLIP models are also [available on Huggingface](https://huggingface.co/docs/transformers/main/en/model_doc/chinese_clip), but here we import the model via [cn_clip package](https://pypi.org/project/cn-clip/) for convenience since its codes are similar to OpenCLIP
  
  To list available models (please see [Model Card](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README_En.md#model-card) provided by ChineseCLIP for more details):
    ```python
    from cn_clip.clip import available_models
    available_models() 
    # ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
    ```
  
  

  
  To load a ChineseCLIP with ResNet-50:
    ```bash
    ... \
    --image-model 'RN50' --image-model-builder 'chineseclip' \
    --text-model 'RN50' --text-model-builder 'chineseclip' \
    --pretrained-image-model --pretrained-text-model \

## Pretrained Uni-modal Models

### Image Backbone

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



### Text Backbone
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
