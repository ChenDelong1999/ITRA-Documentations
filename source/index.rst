.. ITRA documentation master file, created by
   sphinx-quickstart on Fri Nov 18 11:15:19 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the docmentation of ITRA ! 🎈
================================

ITRA (Image Text Representation Alignment) is a codebase for flexible and efficient vision language learning.

TODO list
================================
- CILP finetuning methods (MS-COCO as example)
      - Partially freeze the weighs
            - according to weight type (prams name filter)
            - according to layers
      - MIM and FLIP support
      - 'CLIP Itself is a Strong Fine-tuner' re-implementation (https://arxiv.org/abs/2212.06138, https://discourse.brainpp)
      - Wise-FT re-implementation (https://arxiv.org/abs/2109.01903)

- Method Supports
      - UniCL support
      - SSL Image Augmentations
      - Loading face encoder as image backbone
      - Loading LLMs (OPT, PaLM) as text backbone.cn/t/topic/65205)
- Evaluation Reports
      - Custom evaluations
      - Chinese CLIPs (ImageNet-CN zero-shot, MC-COCO-CN retrieval)
      - MS COCO Retrieval Benchmarks
      - Verify NLP ealuations



.. image:: ../source/Contents/assets/pipeline.png
   :align: right
   :width: 10in

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   Contents/introduction/overview

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Contents/getting-started/install-dependencies
   Contents/getting-started/prepare-pretraining-data
   Contents/getting-started/prepare-downstream-data

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   Contents/user-guide/load-pretrained-multi-modal-weights
   Contents/user-guide/load-pretrained-uni-modal-weights
   Contents/user-guide/training-data
   Contents/user-guide/best-practice

.. toctree::
   :maxdepth: 1
   :caption: Example Usage

   Contents/example-usage/clip-pretraining
   Contents/example-usage/clip-finetuning
   Contents/example-usage/eval-only


.. toctree::
   :maxdepth: 1
   :caption: Experiment Reports

   Contents/experiment-reports/report_1