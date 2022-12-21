# Load Pretrained Multi-modal Weights

## From `OpenCLIP`

[OpenCLIP](https://github.com/mlfoundations/open_clip) (v2.0.2) is an open source implementation of [OpenAI's CLIP](https://github.com/openai/CLIP) (Contrastive Language-Image Pre-training). To check all supported model architecture and pre-trained weights, run:

```python
import open_clip
open_clip.list_pretrained()
# [('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN50', 'cc12m'), ('RN50-quickgelu', 'openai'), ('RN50-quickgelu', 'yfcc15m'), ('RN50-quickgelu', 'cc12m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'), ('RN101-quickgelu', 'openai'), ('RN101-quickgelu', 'yfcc15m'), ('RN50x4', 'openai'), ('RN50x16', 'openai'), ('RN50x64', 'openai'), ('ViT-B-32', 'openai'), ('ViT-B-32', 'laion400m_e31'), ('ViT-B-32', 'laion400m_e32'), ('ViT-B-32', 'laion2b_e16'), ('ViT-B-32', 'laion2b_s34b_b79k'), ('ViT-B-32-quickgelu', 'openai'), ('ViT-B-32-quickgelu', 'laion400m_e31'), ('ViT-B-32-quickgelu', 'laion400m_e32'), ('ViT-B-16', 'openai'), ('ViT-B-16', 'laion400m_e31'), ('ViT-B-16', 'laion400m_e32'), ('ViT-B-16-plus-240', 'laion400m_e31'), ('ViT-B-16-plus-240', 'laion400m_e32'), ('ViT-L-14', 'openai'), ('ViT-L-14', 'laion400m_e31'), ('ViT-L-14', 'laion400m_e32'), ('ViT-L-14', 'laion2b_s32b_b82k'), ('ViT-L-14-336', 'openai'), ('ViT-H-14', 'laion2b_s32b_b79k'), ('ViT-g-14', 'laion2b_s12b_b42k'), ('roberta-ViT-B-32', 'laion2b_s12b_b32k'), ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'), ('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k')]
```

To load the official pretrained CLIP (ResNet-50):

```bash
--image-model 'RN50' --image-model-builder 'openclip' \
--text-model 'RN50' --text-model-builder 'openclip' \
--pretrained-image-model --pretrained-text-model \
```

Optionally, you can load CLIP models pretrained by OpenCLIP instead of OpenAI by specifying `--image-model-tag` and `--text-model-tag`. For example, to load the [ViT-H-14 pretrained on LAION-2B](https://github.com/mlfoundations/open_clip#vit-h14-224x224):

```bash
--image-model 'ViT-H-14' --image-model-builder 'openclip' --image-model-tag 'laion2b_s32b_b79k' \
--text-model 'ViT-H-14' --text-model-builder 'openclip'  --text-model-tag 'laion2b_s32b_b79k' \
--pretrained-image-model --pretrained-text-model \
```

## From `ChineseCLIP` 

[ChineseCLIP](https://github.com/OFA-Sys/Chinese-CLIP) (v1.4) is the Chinese version of CLIP. We use a large-scale Chinese image-text pair dataset (~200M) to train the model, and we hope that it can help users to conveniently achieve image representation generation, cross-modal retrieval and zero-shot image classification for Chinese data. This repo is based on OpenCLIP project.
  
The ChineseCLIP models are also [available on Huggingface](https://huggingface.co/docs/transformers/main/en/model_doc/chinese_clip), but here we import the model via [cn_clip package](https://pypi.org/project/cn-clip/) for convenience since its codes are similar to OpenCLIP
  
To list available models (please see [Model Card](https://github.com/OFA-Sys/Chinese-CLIP/blob/master/README_En.md#model-card) provided by ChineseCLIP for more details):

```python
from cn_clip.clip import available_models
available_models() 
# ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
```
    
To load a ChineseCLIP with ResNet-50:
```bash
--image-model 'RN50' --image-model-builder 'chineseclip' \
--text-model 'RN50' --text-model-builder 'chineseclip' \
--pretrained-image-model --pretrained-text-model \
```


## From `Taiyi-CLIP` 

[Taiyi-CLIP](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese) （封神榜-太乙）employs [chinese-roberta-wwm](https://huggingface.co/hfl/chinese-roberta-wwm-ext) for the language encoder, and apply the ViT-B-32 in CLIP for the vision encoder. They freeze the vision encoder and tune the language encoder to speed up and stabilize the pre-training process. Moreover, they apply [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/) dataset (100M) and [Zero](https://zero.so.com/) dataset (23M) as the pre-training datasets. See their [documentations](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E5%A4%AA%E4%B9%99%E7%B3%BB%E5%88%97/index.html) for details.

  There are two CLIP models available via `Taiyi-CLIP`: `Taiyi-CLIP-Roberta-102M-Chinese` ([doc](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E5%A4%AA%E4%B9%99%E7%B3%BB%E5%88%97/Taiyi-CLIP-Roberta-102M-Chinese.html)) and `Taiyi-CLIP-Roberta-large-326M-Chinese` ([doc](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E5%A4%AA%E4%B9%99%E7%B3%BB%E5%88%97/Taiyi-CLIP-Roberta-large-326M-Chinese.html)). These two models are trained by [Locked Image Tuning (LiT)](https://arxiv.org/abs/2111.07991) on the ViT-B-32 and ViT-L-14 of OpenAI's CLIP. Therefore, to load these model:

```bash
# Taiyi-CLIP-Roberta-102M-Chinese
--image-model 'ViT-B-32' --image-model-builder 'openclip' \
--text-model 'IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese' --text-model-builder 'huggingface' \
--pretrained-image-model --pretrained-text-model \

# Taiyi-CLIP-Roberta-102M-Chinese
--image-model 'ViT-L-14' --image-model-builder 'openclip' \
--text-model 'IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese' --text-model-builder 'huggingface' \
--pretrained-image-model --pretrained-text-model \
```