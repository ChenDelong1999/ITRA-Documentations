# Load Pretrained Uni-modal Weights

## Image Backbone

### From `Torchvision`

To check all supported model architecture and pretrained weigths, run the following command or see [this page](https://pytorch.org/vision/0.12/models.html)  (v0.12).

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


### From `Torch Hub`

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

---

## Text Backbone

### From HuggingFace🤗Transformers

For more details, see [HuggingFace Transformers](https://huggingface.co/docs/transformers). Currently, only 'from pretrained' mode is supported (i.e., you cannot train a huggingface transformer from scratch now). Standard models like BERT/RoBERTa are supported, but whether other models are also supported is not sure...

### From Sentence Transformers

The [Sentence Transformers](https://www.sbert.net) liberary provides powerfull sentence embeddings. Please see [pretrained models](https://www.sbert.net/docs/pretrained_models.html) for more detials. Loading sentence transformers via huggingface and specify `--text-pooler='mean'` is recommended, though it is also supported to load the model via sentence transformer:

```bash
# recommended: 
--text-model-builder 'huggingface'  --text-model 'sentence-transformers/all-mpnet-base-v2' --text-pooler='mean' 
# not recommended:
--text-model-builder 'sbert'  --text-model 'all-mpnet-base-v2' 
```

However, it seems that word embedding models ([GloVe](https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d) and [Komninos](https://huggingface.co/sentence-transformers/average_word_embeddings_komninos)) in sentence-transformers cannot be loaded via huggingface.
