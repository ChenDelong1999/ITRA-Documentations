
# Use Adapters
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

- Projection Head Adapters
  - Linear projection head
  - [DINO MLP Head](https://github.com/facebookresearch/dino/blob/cb711401860da580817918b9167ed73e3eef3dcf/vision_transformer.py#L257) (optionally with a prototype layer in the last)

