# Training Data

## Reading Pre-training Data from CSV file

```bash
# RSCID
--train-data '/data/Datasets/RSICD/csv/rsicd_test.csv' \
```


## Episodic Training

```bash
--dataset-size 14000000 --episode-size 4000000 --train-data 'cache/yfcc_nori.csv' --nori-dataset\
--epochs 28 --save-frequency 28 --batch-size 64 --workers 8 \
```


## Use NORI (for Megvii Useres)

### Nori Speed-up

Conceptual Captions 3M

```bash
nori speedup 's3://chendelonghahab/datasets/ConceptualCaption3M/nori_CC270w.csv' --on --replica=2
```

YFCCM-14M
```zsh
for ((i=0;i<=100;i++)) {
    echo 'Processing nori part '$i'/100...'
    nori speedup 's3://yzq/mmsl_datasets/YFCC15M/yfcc15m_'$i'.nori' --on --replica=2
}
```
