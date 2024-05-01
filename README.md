## [ICMI 2023] Make Your Brief Stroke Real and Stereoscopic: 3D-Aware Simplified Sketch to Portrait Generation


# Table of Content
- [News](#news)
- [Installation](#step-by-step-installation-instructions)
- [Prepare Data](#prepare-data)
- [Pretrained Model](#pretrained-model)
- [Testing](#testing)
- [License](#license)
- [Acknowledgements](#acknowledgements)


# News
- [2024/07]: Paper is accepted on ICMI 2023 (https://dl.acm.org/doi/abs/10.1145/3577190.3614106)
- [2023/02]: Paper is on [arxiv](https://arxiv.org/abs/2302.06857)
- [2024/05]: Demo and code released.


# Step-by-step Installation Instructions

**a. Create a conda virtual environment and activate it.**
It requires python >= 3.7 as base environment.
```shell
conda create -n sssp python=3.7 -y
conda activate sssp
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch==1.10.0 torchvision==0.8.2 -c pytorch -c conda-forge
```

**c. Install other dependencies.**
We simply freeze our environments. Other environments might also works. Here we provide requirements.txt file for reference.
```shell
pip -r requirements.txt
```


# Prepare Data
- Download Cropped FFHQ Data ()


# Pretrained Model
Download [Pretrained model] ()


# Testing
```
bash experiments/test.sh sketch_w_128_quanti
```


## Citation
If you find our paper and code useful for your research, please consider citing:
```bibtex
@misc{sun2023make,
  title={Make Your Brief Stroke Real and Stereoscopic: 3D-Aware Simplified Sketch to Portrait Generation},
  author={Sun, Yasheng and Wu, Qianyi and Zhou, Hang and Wang, Kaisiyuan and Hu, Tianshu and Liao, Chen-Chieh and Miyafuji, Shio and Liu, Ziwei and Koike, Hideki},
  booktitle={Proceedings of the 25th International Conference on Multimodal Interaction},
  pages={388--396},
  year={2023}
}
```