# Related Work

## OpenCLIP (LAION AI)

[[github]](https://github.com/mlfoundations/open_clip) | [[pypi]](https://pypi.org/project/open-clip-torch/)

[[paper]](https://arxiv.org/abs/2212.07143): Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, Jenia Jitsev. **Reproducible scaling laws for contrastive language-image learning**. _ArXiv preprint_. 

Welcome to an open source implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pre-training).

The goal of this repository is to enable training models with contrastive image-text supervision, and to investigate their properties such as robustness to distribution shift. Our starting point is an implementation of CLIP that matches the accuracy of the original CLIP models when trained on the same dataset.
Specifically, a ResNet-50 model trained with our codebase on OpenAI's [15 million image subset of YFCC](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md) achieves **32.7%** top-1 accuracy on ImageNet. OpenAI's CLIP model reaches **31.3%** when trained on the same subset of YFCC. For ease of experimentation, we also provide code for training on the 3 million images in the [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download) dataset, where a ResNet-50x4 trained with our codebase reaches 22.2% top-1 ImageNet accuracy.

We further this with a replication study on a dataset of comparable size to OpenAI's, [LAION-400M](https://arxiv.org/abs/2111.02114), and with the larger [LAION-2B](https://laion.ai/blog/laion-5b/) superset. In addition, we study scaling behavior in a paper on [reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143).

---

## LAVIS - **LA**nguage **VIS**ion (Salesforce Research)

[[github]](https://github.com/salesforce/LAVIS) | [[doc]](https://opensource.salesforce.com/LAVIS)

[[paper]](https://arxiv.org/abs/2209.09019): Dongxu Li, Junnan Li, Hung Le, Guangsen Wang, Silvio Savarese, Steven C.H. Hoi. **LAVIS: A Library for Language-Vision Intelligence**. _ArXiv preprint_. 

LAVIS is a Python deep learning library for LAnguage-and-VISion intelligence research and applications. This library aims to provide engineers and researchers with a one-stop solution to rapidly develop models for their specific multimodal scenarios, and benchmark them across standard and customized datasets.
It features a unified interface design to access
- **10+** tasks
(retrieval, captioning, visual question answering, multimodal classification etc.);
- **20+** datasets (COCO, Flickr, Nocaps, Conceptual
Commons, SBU, etc.);
- **30+** pretrained weights of state-of-the-art foundation language-vision models and their task-specific adaptations, including [ALBEF](https://arxiv.org/pdf/2107.07651.pdf),
[BLIP](https://arxiv.org/pdf/2201.12086.pdf), [ALPRO](https://arxiv.org/pdf/2112.09583.pdf), [CLIP](https://arxiv.org/pdf/2103.00020.pdf).

---

## MMF (Facebook AI Research)

[[homepage]](https://mmf.sh/) | [[github]](https://github.com/facebookresearch/mmf) | [[doc]](https://mmf.sh/docs/)

MMF is a modular framework for vision and language multimodal research from Facebook AI Research. MMF contains reference implementations of state-of-the-art vision and language models and has powered multiple research projects at Facebook AI Research. See full list of project inside or built on MMF [here](https://mmf.sh/docs/notes/projects).

MMF is powered by PyTorch, allows distributed training and is un-opinionated, scalable and fast. Use MMF to **_bootstrap_** for your next vision and language multimodal research project by following the [installation instructions](https://mmf.sh/docs/). Take a look at list of MMF features [here](https://mmf.sh/docs/getting_started/features).

MMF also acts as **starter codebase** for challenges around vision and
language datasets (The Hateful Memes, TextVQA, TextCaps and VQA challenges). MMF was formerly known as Pythia. The next video shows an overview of how datasets and models work inside MMF. Checkout MMF's [video overview](https://mmf.sh/docs/getting_started/video_overview).

---

## X-modaler (JD AI Research)

[[github]](https://github.com/YehLi/xmodaler) | [[doc]](https://xmodaler.readthedocs.io/en/latest/)

[[paper]](https://arxiv.org/abs/2108.08217): Yehao Li, Yingwei Pan, Jingwen Chen, Ting Yao, Tao Mei. **X-modaler: A Versatile and High-performance Codebase for Cross-modal Analytics**. _ACMMM Open Source Software Competition_. 

X-modaler is a versatile and high-performance codebase for cross-modal analytics. This codebase unifies comprehensive high-quality modules in state-of-the-art vision-language techniques, which are organized in a standardized and user-friendly fashion.

---

## TorchMultimodal (Facebook AI Research)
  
[[github]](https://github.com/facebookresearch/multimodal)

TorchMultimodal is a PyTorch library for training state-of-the-art multimodal multi-task models at scale. It provides:
- A repository of modular and composable building blocks (models, fusion layers, loss functions, datasets and utilities).
- A repository of examples that show how to combine these building blocks with components and common infrastructure from across the PyTorch Ecosystem to replicate state-of-the-art models published in the literature. These examples should serve as baselines for ongoing research in the field, as well as a starting point for future work.

As a first open source example, researchers will be able to train and extend FLAVA using TorchMultimodal.


## Multimodal-Toolkit (Georgian)

[[github]](https://github.com/georgian-io/Multimodal-Toolkit) | [[doc]](https://multimodal-toolkit.readthedocs.io/en/latest/index.html)

[[paper]](https://aclanthology.org/2021.maiworkshop-1.10.pdf) Ken Gu, Akshay Budhkar. **Multimodal-Toolkit: A Package for Learning on Tabular and Text Data with Transformers**. _Third Workshop on Multimodal Artificial Intelligence_.

A toolkit for incorporating multimodal data on top of text data for classification
and regression tasks. It uses HuggingFace transformers as the base model for text features.
The toolkit adds a combining module that takes the outputs of the transformer in addition to categorical and numerical features
to produce rich multimodal features for downstream classification/regression layers.
Given a pretrained transformer, the parameters of the combining module and transformer are trained based
on the supervised task. For a brief literature review, check out the accompanying [blog post](https://medium.com/georgian-impact-blog/how-to-incorporate-tabular-data-with-huggingface-transformers-b70ac45fcfb4) on Georgian's Impact Blog. 



## 贵司公开 Codebase

- https://github.com/megvii-research/mdistiller
- https://github.com/PyRetri/PyRetri