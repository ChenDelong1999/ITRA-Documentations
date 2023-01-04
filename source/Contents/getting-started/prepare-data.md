# Prepare Data

## Image-text Pairs Dataset from `CSV` file

This codebase reads a `CSV` file (separated by `\t`) with two columns: a path to an image (`filepath` by default), and a text caption (`title` by default). 

| filepath          | title                      |
|-------------------|----------------------------|
| path/to/image.jpg | A very typical bus station |
| ...               | ...                        |

Specifying `--train-data 'path/to/your/csvfile.csv'` enables training a model on the dataset, and specifying `--retrieval-data 'path/to/your/csvfile.csv'` and set `--retrieval-frequency` > 0 to perform retrieval evaluation on the dataset.

The script `itra/utils/gather_cc.py` will collect the [Conceptual Captions](https://github.com/google-research-datasets/conceptual-captions) (CC3M) dataset. First, download the Conceptual Captions URLs from [here](https://ai.google.com/research/ConceptualCaptions/download), then run the following script:

```bash
python3 itra/utils/gather_cc.py path/to/Train_GCC-training.tsv
```

```eval_rst
.. note::
    As mentioned in our ProtoCLIP paper, the CC3M dataset was made public by Google in 2018. As noted in our paper, the number of accessible images keeps drooping due to expired image links. This issue is raised by several recent works. In this work, since we can only collect 2,643,718 images (concurrent to our work ProtoCLIP, CyCLIP collected 2,631,703 images), we randomly sample a 2,500,000 subset (75\% of full CC3M) from them to train our ProtoCLIP. Considering the dropping accessibility of image links in Conceptual Captions, we call for the use of this dataset size (2.5M) in future benchmarking for better comparability.
```

```eval_rst
.. important::
    The requirement of CC3M validation data of OpenCLIP is removed in this codebase. To perform retrieval evaluation, please use the ``--retrieval-data`` argument instead. The `webdataset` is no longer supported in this codebase.
```

## MS COCO Captions dataset

To use MS COCO 2017 Captions dataset, download it to `--datasets-dir` and specifying `--train-data 'mscoco_captions'` or `--retrieval-data 'mscoco_captions'`.

```bash
<--datasets-dir>
    └──coco2017
        ├── annotations
        ├── train2017 
        └── val2017 
```

The dataset contains 118k train images and 5k text images, and each image has 4-5 captions. When using the training images, the total samples per epoch is set to 118k, and we chose one caption randomly when calling the `__getitem__` function. 


## Image Classification Dataset

Add your dataset into `itra/data/classification_datasets.py` and add your dataset name (e.g., 'YourCustomDataset') to `AVALIABLE_CLASSIFICATION_DATASETS`. Then you can use this dataset via `--train-data 'YourCustomDataset'`.


## SentEval Datasets

Codes for SentEval evaluation are modified from [SimCSE](https://github.com/princeton-nlp/SimCSE#evaluation).

```bash
cd <--dataset-dir>
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/senteval.tar
tar xvf senteval.tar
```

```eval_rst
.. todo ::
    - MS MARCO
    - wiki sections
```



## EVEVATER Image Classification Datasets

[EVEVATER Image Classification Toolkit](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC) (Elevater_Toolkit_IC) implemeted standarlized evaluations of vision language models. It covers zero-shot classification, few- / full-shot linear probing, and fully fine tuning on 20 datasets. See paper "*[ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models](https://arxiv.org/abs/2204.08790), NeurIPS 2022 Datasets and Benchmarks Track*" for more details.

We have included Elevater_Toolkit_IC in our codebase (in `itra/evaluation/vision_benchmark`). We have registered new models ([clip_zeroshot_eval.py]((src/training/evaluations/vision_benchmark/models/clip_zeroshot_eval.py)) and [cls_linear_or_ft_eval.py]((itra/evaluation/vision_benchmark/models/cls_linear_or_ft_eval.py))) following the official instructions. To ensure compatibility, we have made some modifications based on the official Elevater_Toolkit_IC codes at commit `9d39620`, so DO NOT install an Elevater_Toolkit_IC in the environment for this codebase.

To get started first download all dataset following [this repo](https://github.com/Computer-Vision-in-the-Wild/DataDownload). The downloaded datasets takes about 41Gb storage, and the folder structure should be: 


```bash
.../datasets
└── classification
    ├── caltech_101_20211007
    │   ├── labels.txt
    │   ├── test.txt
    │   ├── test.zip
    │   ├── train.txt
    │   └── train.zip
    ├── cifar100_20200721
    │   ├── labels.txt
    │   ├── test_images.txt
    │   ├── test_images.zip
    │   ├── train_images.txt
    │   └── train_images.zip
    ...
    └── voc2007_20211007
        ├── labels.txt
        ├── test_ic.txt
        ├── test.zip
        ├── train_ic.txt
        ├── train.zip
        └── val_ic.txt

21 directories, 115 files
```

## NORI Datasets on OSS (for Megvii Useres)

- To use Conceptual Captions 3M: `--train-data 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC2716261.csv'`

```bash
# Nori Speed-up Commands
nori speedup 's3://chendelong/datasets/ConceptualCaption3M/CC_3M.nori' --on --replica=2
nori speedup 's3://chendelonghahab/datasets/ConceptualCaption3M/CC2.6M-CC2M.nori/' --on --replica=2
```

- To use YFCCM-14M: `--train-data 's3://chendelonghahab/datasets/YFCC/YFCC_cleaned_nori.csv' `

```bash
# zsh
# Nori Speed-up Commands
for ((i=0;i<=100;i++)) {
    echo 'Processing nori part '$i'/100...'
    nori speedup 's3://yzq/mmsl_datasets/YFCC15M/yfcc15m_'$i'.nori' --on --replica=2
}
```
