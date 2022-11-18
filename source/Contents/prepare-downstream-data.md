
# Prepare Downstream Data
- **Zero-shot Classification**. The preprocessed zero-shot datasets can be downloaded from [CLOOB](https://github.com/ml-jku/cloob#downstream-tasks).

- **Linear Probing**. We perform linear evaluation on ImageNet, CIFAR10, CIFAR100, and STL10. You need to download the full [ImageNet-1k](https://image-net.org/download.php) dataset manually. The later three datasets are integrated into `torchvision` and will be downloaded automatically.

- **Image-text Retrieval**. We implement zero-shot image-text retrieval on MS-COCO. Since we do not perform fine-tuning, only the validation split (`/val2017`) is required here.

    
    ```
    # All downstream datasets shall be stored to <YOUR DATASET ROOT> dictionary:
    <YOUR DATASET ROOT>
        ├── imagenet
        │   ├── train
        │   └── test  
        ├── birdsnap
        │   └── test
        ├── country211
        │   └── test
        ├── flowers102
        │   └── test
        ├── gtsrb
        │   └── test
        ├── stanford_cars
        │   └── test
        ├── ucf101
        │   ├── testlist01
        │   ├── testlist02
        │   └── testlist03   
        └── coco2017
           ├── annotations
           └── val2017 
    ```

- **STS**
https://github.com/princeton-nlp/SimCSE#evaluation

- **MS MARCO**

- **wiki sections**

- EVEVATER Image Classification Toolkit

    [EVEVATER Image Classification Toolkit](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC) (Elevater_Toolkit_IC) implemeted standarlized evaluations of vision language models. It covers zero-shot classification, few- / full-shot linear probing, and fully fine tuning on 20 datasets. See paper "*[ELEVATER: A Benchmark and Toolkit for Evaluating Language-Augmented Visual Models](https://arxiv.org/abs/2204.08790), NeurIPS 2022 Datasets and Benchmarks Track*" for more details.

    We have included Elevater_Toolkit_IC in our codebase (in `src/training/evaluations/vision_benchmark`). We have registered new models ([clip_zeroshot_eval.py]((src/training/evaluations/vision_benchmark/models/clip_zeroshot_eval.py)) and [cls_linear_or_ft_eval.py]((src/training/evaluations/vision_benchmark/models/cls_linear_or_ft_eval.py))) following the official instructions. To ensure compatibility, we have made some modifications based on the official Elevater_Toolkit_IC codes at commit `9d39620`, so DO NOT install an Elevater_Toolkit_IC in the environment for this codebase.

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
