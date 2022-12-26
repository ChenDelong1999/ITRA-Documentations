
![pipeline](./assets/pipeline.png "pipelinee")

# About This Codebase
ITRA is a codebase for flexible and efficient Image Text Representation Alignment.

## Supported Methods

### Model Builder
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [Torchvision (v0.12)](https://pytorch.org/vision/0.12/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html)
- [Adapter-Transformers](https://github.com/adapter-hub/adapter-transformers)

### Training Objectives
- CLIP: InfoNCE, ProtoCLIP
- Self-supervised KD: RKD, SEED, CompRess, ProtoCPC, SimReg
- VICReg, BarlowTwins, DINO

### Downstream Evaluation
- Image classification: zero-shot, linear/k-NN, and clustering evaluation (AMI, NMI) (from [ProtoCLIP](https://github.com/megvii-research/protoclip))
- [EVEVATER Image Classification Toolkit](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC) on 20 datasets
- Image-text retrieval on MS-COCO dataset
- Sentence embeddings ([SentEval](https://github.com/facebookresearch/SentEval))
- Passage retrieval on MS-MARCO and Wiki Sections
- Word embeddings: RG65, Simlex999, WordSim353
- Zero-shot VQA ([TAP-C](https://arxiv.org/abs/2203.07190)) and visual entailment 


