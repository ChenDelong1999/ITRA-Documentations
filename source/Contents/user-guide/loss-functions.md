
# Loss Functions

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

