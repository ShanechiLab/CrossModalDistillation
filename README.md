# Cross-modal knowledge distillation 
Cross-modal knowledge distillation from multi-session spike models to LFP models to develop LFP models with enhanced representational power. 

## Setting up the environment
You can set up a Python 3.9 environment via `conda create -n <env_name> python=3.9`. Then, after activating your virtual environment, please install the required packages and `cross_modal_distillation` package through 

- `pip install torch==2.4.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121`
- `pip install -e .`

## Data and configs
- The data configs are located under `cross_modal_distillation/data/configs` (for final datasets used for model training, combines individual dataset configs) and `cross_modal_distillation/data/configs/single` (for individual datasets). 
- The default data location is denoted under `base.yaml` (under `cross_modal_distillation/data/configs/single`) config as `./results/data`. If you prefer a different location, please make sure to change it.
- Dataset download operations are automatized for `MakinRT` dataset, but not for the `FlintCO` dataset. For that dataset, please download the raw dataset from https://crcns.org/data-sets/movements/dream/downloading-dream, and place the downloaded `.mat` files (e.g., `Flint_2012_e1.mat`) under the `<root_save_dir>/FlintCO/raw` directory (note that the default `root_save_dir` is `./results/data`).
- In our experiments, we used 5 second segments which were generated from 10 second segments by chunking. We followed this logic to ensure same train/val/test splits across different segment lenghts, which was a part of our design and experimentation process. These steps can also be found in `inference_generalization.ipynb` notebook.  

## Checkpoints
We provide multiple model checkpoints for reproducing `Fig. 5` in `Section 4.3`: 
- `ms_lfp.ckpt`: This is the MS-LFP model pretrained on the LFP signals of 34 recording sessions across 2 subjects. This model included all sessions shown in `Fig. 5` in its pretraining dataset. Also, this is the MS-LFP model whose performance was also shown in `Fig. 3` and other figures, after fine-tuning on the LFP signals of the sessions shown if that session was held-out in the pretraining.
- `distilled_lfp_monkeyI_20160622_01.ckpt`: This is the Distilled LFP model trained on the LFP signals of `Monkey I`'s `20160622_01` recording session.
- `distilled_lfp_monkeyC_e1_1.ckpt`: Same as `distilled_lfp_monkeyI_20160622_01.ckpt`, but for `Monkey C`'s `e1_1` recording session. 

## Inference generalization
Please follow the notebook we provided in `inference_generalization.ipynb` to reproduce the generalization results shown in `Fig. 5` (`Section 4.3`). Overall, the steps followed in that notebook are: 

- Generating 10 second and 5 second LFP datasets. 
- Building and loading the MS-LFP checkpoint provided.
- Performing inference and decoding using the loaded MS-LFP model. These results serve as the **baseline** for evaluating the generalization performance of the Distilled LFP models. Note that while the recording sessions used for generalization were included in the pretraining dataset of the loaded MS-LFP model, these generalization sessions were **entirely excluded** from the distillation process as well as from the spike teacher model training.
- Building and loading the Distilled LFP models trained on recording sessions `20160622_01` and `e1_1`. 
- Perform inference and decoding with the loaded Distilled LFP models by passing the LFP signals from the generalization sessions as if they originated from the recording sessions used during distillation. 

Please check out our manuscript for full results and further details.

## Publication:
[Erturk, E., Hashemi, S., Shanechi, M. M. Cross-Modal Representational Knowledge Distillation for Enhanced Spike-informed LFP Modeling. In Advances in Neural Information Processing Systems 2025.](https://openreview.net/forum?id=hT7Nj7SAQb)

## Licence:
Copyright (c) 2025 University of Southern California <br />
See full notice in [LICENSE.md](LICENSE.md) <br />
Eray Erturk, Saba Hashemi, and Maryam M. Shanechi <br />
Shanechi Lab, University of Southern California <br />