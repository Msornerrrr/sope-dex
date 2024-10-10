# Singulating Object in Packed Environment

<div align=center>
<img src="assets/image_folder/sope-dex.gif" border=0 width=75%>
</div>

## Table of Content
- [Overview](#overview)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)
- [Citations](#citations)
- [License](#license)

## Overview
This repository is the implementation code of the paper "Learning to Singulate Objects in Packed
Environments using a Dexterous Hand" ([Paper](https://arxiv.org/abs/2409.00643), [Website](https://sope-dex.github.io/)) by Hao Jiang, Yuhai Wang*, Hanyang Zhou*, and Daniel Seita. In this repo, we provide our full implementation code of simulation.

## Installation
* python 3.8
```
conda create -n sopedex python=3.8
conda activate sopedex
```

* IsaacGym (tested with `Preview Release 3/4` and `Preview Release 4/4`). Follow the [instruction](https://developer.nvidia.com/isaac-gym) to download the package.
```
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
(test installation) python examples/joint_monkey.py
```
If failed, you can try to add the path to your isaacgym python library at the start of the `train.py` file
```
sys.path.append('/home/$USER/isaacgym_project/isaacgym/python')
```
(change the path according to the location of isaacgym in your machine)

* Singulating Object in Packed Environment
```
git clone https://github.com/Msornerrrr/sope-dex.git
cd sope-dex
pip install -e .
```

* Test your installation using this code.
```
cd dexteroushandenvs/
bash scripts/experiment.sh
```
* Trained checkpoint. Download from [Link](https://drive.google.com/file/d/1_D6juQDMIhkEreqRCRpun3UUq5n5l_f1/view?usp=drive_link).


## Training

To train the model, run this line in `dexteroushandenvs` folder:

```
bash scripts/train.sh
```

The trained model will be saved to `logs` folder

## Testing
To load a trained model and only perform inference (no training)
```
bash scripts/test.sh
```
you can specify which trained model to use by changing `--model_dir` to use the corresponding path.


## Acknowledgement
We thank the list of contributors from the [Dynamic Handover](https://github.com/cypypccpy/dynamic_handover), [Sequential Dexterity](https://github.com/sequential-dexterity/SeqDex), and [Bi-DexHands](https://github.com/PKU-MARL/DexterousHands).

## Citations
Please cite [SOPE DEX](https://sope-dex.github.io/) if you use this repository in your publications:
```
@inproceedings{Hao2024SopeDex,
  author = {Hao Jiang and Yuhai Wang and Hanyang Zhou and Daniel Seita},
  title = {{Learning to Singulate Objects in Packed Environments using a Dexterous Hand}},
  booktitle = {International Symposium on Robotics Research (ISRR)},
  year = {2024}
}
```

## License
Licensed under the [MIT License](LICENSE)
