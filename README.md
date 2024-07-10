# SIAB
Implementation of the paper *The Devil is in the Statistics: Mitigating and Exploiting Statistics Difference for Generalizable Semi-supervised Medical Image Segmentation*.

## Getting Started

### Installation

We use `miniconda` to manage the environment.
After cloning this repo and [installing miniconda](https://docs.anaconda.com/miniconda/miniconda-install/), create the environment by simply running:

```bash
git clone https://github.com/qiumuyang/SIAB.git

cd SIAB

conda env create -f env.yml
```


### Data Preparation

#### Prostate

Data is available at [SAML](https://liuquande.github.io/SAML).
Please download the data (in 3D) and preprocess it as `utils/preprocess_prostate.py` shows (to 2D).

```bash
# /path/to/raw/prostate includes directories for 6 domains
python utils/preprocess_prostate.py /path/to/raw/prostate --output /path/to/preprocessed/prostate
```

#### Fundus

Data is available at [DoFE](https://github.com/emma-sjwang/Dofe).
Preprocess it to keep the ROIs only.

```bash
# /path/to/raw/fundus includes directories for 4 domains
python utils/preprocess_fundus.py /path/to/raw/fundus --output /path/to/preprocessed/fundus
```

#### M&Ms

Data and preprocessing code are available at [DGNet](https://github.com/xxxliu95/DGNet).

#### File Structure

After preprocessing, the data directory would look like this:

```bash
data
├── prostate_processed
│   ├── BIDMC
│   ├── BMC
│   ├── HK
│   └── ... (other domains)
├── fundus_processed
│   ├── Domain1
│   └── ... (other domains)
└── mms_processed
    ├── mnms_split_2D_data
    ├── mnms_split_2D_mask
    └── mnms_split_2D_re
```

Then, modify `configs/{prostate,fundus,mnms}.yaml` to set `data_root` to corresponding paths.

## Training

You can train the model for one **single domain** with the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/prostate.yaml \
    --train-config configs/train.yaml \
    --save-path outputs/prostate/0.3/A \
    --seed 0 \
    --domain 0 \
    --ratio 0.3
```

Also, you can train the model for all domains **one by one** or **in parallel** with the following command:

```bash
bash scripts/train_seq.sh train.py prostate 0 0.3 train
bash scripts/train_par.sh train.py prostate 0,1,2 0.3 train
```

## Evaluation

```bash
python infer.py <checkpoint.pth> \
    --config configs/prostate.yaml \
    --seed 0 \
    --domain 0
```

## Acknowledgement

Our implementation is partially based on [UniMatch](https://github.com/LiheYoung/UniMatch).
