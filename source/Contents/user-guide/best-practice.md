
# Best Practice 
### **From Adapter-Transformers**. 

The [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers) liberary enables Delta-tuning on popular huggingface transformers. See [Model Overview](https://docs.adapterhub.ml/model_overview.html) for available adaptations, and see the [Docs](https://docs.adapterhub.ml/) and [AdapterHub](https://adapterhub.ml/) for more details.

    We have made the following adapters available in this codebase:

|                                                Adapter                                                |     args.adapter     | Params (M) | Params (%) | STS Benchmark | ImageNet Zero-shot Accuracy | MSCOCO Retrieval Mean Recall |
|:-----------------------------------------------------------------------------------------------------:|:--------------------:|------------|------------|---------------|----------------------------|------------------------------|
|                    [Compacter](https://docs.adapterhub.ml/overview.html#compacter)                    |        `dummy`       | 0.06       | 0.05%      | 0.7474        | 24.48                      | 38.73                        |
|                        [(IA)^3](https://docs.adapterhub.ml/overview.html#ia-3)                        |     `ia3_adapter`    | 0.06       | 0.05%      | 0.6576        | 19.23                      | 31.90                        |
|                         [LoRA](https://docs.adapterhub.ml/overview.html#lora)                         |    `lora_adapter`    | 0.30       | 0.27%      | 0.7514        | 25.02                      | 40.58                        |
|         [Bottleneck   adapters](https://docs.adapterhub.ml/overview.html#bottleneck-adapters)         | `bottleneck_adapter` | 1.79       | 1.61%      | 0.7449        | 26.15                      | 41.85                        |
| [Language   Adapters](https://docs.adapterhub.ml/overview.html#language-adapters-invertible-adapters) |    `lang_adapter`    | 1.19       | 1.08%      | 0.7405        | 26.71                      | 42.39                        |
|               [Prefix   Tuning](https://docs.adapterhub.ml/overview.html#prefix-tuning)               |    `prefix_tuning`   | 9.88       | 8.28%      | 0.7303        | 26.00                      | 41.31                        |
|                      [UniPELT](https://docs.adapterhub.ml/overview.html#unipelt)                      |       `unipelt`      | 11.09      | 9.20%      | 0.7441        | 26.89                      | 43.45                        |
|      [Mix-and-Match   Adapters](https://docs.adapterhub.ml/overview.html#mix-and-match-adapters)      |     `mam_adapter`    | 22.50      | 17.05%     | 0.7503        | 29.61                      | 45.82                        |

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

