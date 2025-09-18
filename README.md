# datanalytics-riscv

This repository contains the PyTorch and TorchVision compiled wheels compatible with RISC-V (RV64GCV), as well as other workloads used within the project's framework.

One can find a set of workloads (VGG19 for now) to test pytorch and torchvision packages functionality. It has to be noted that this repository serves as a proof of concept and that the workloads are focused on testing pytorch and showing that it can be already used on RISC-V machines with vector instructions to train models. Therefore, one will see that the vgg19 and googlenet models might not provide a production-ready efficiency and performance.

Here you can find a set of steps to set up your environment with TorchVision and Pytorch for RV64GCV machines:

The wheels can be found on the release tagged as torch-rvv-0.1. These are precompiled PyTorch and TorchVision wheels for riscv64 with vector instructions (RV64GCV).

- PyTorch version: 2.9.0a0+gita714437
- TorchVision version: 0.20.1a0+3ac97aa
- Python version: 3.12.3
- Tested on: Linux 6.6.36, riscv64

### Set up and Install process

Download the wheels from the release assets and install with pip (it is recommended to install it on a python virtual environment of yours):

```bash
source ~/<your-venv>/bin/activate # Recommended

pip install https://github.com/vitamin-v-software/datanalytics-rsicv/releases/download/torch-rvv-0.1/torch-2.9.0a0+gita714437-cp312-cp312-linux_riscv64.whl
pip install https://github.com/vitamin-v-software/datanalytics-rsicv/releases/download/torch-rvv-0.1/torchvision-0.20.1a0+3ac97aa-cp312-cp312-linux_riscv64.whl
```

By executing the following command, you should already obtain the above PyTorch and TorchVision versions, proving that the packages are installed on your venv (if using one):

```bash
# PyTorch test
python3 -c "import torch; print('Torch version:', torch.__version__)" 

# TorchVision test
python3 -c "import torchvision; print('Torchvision version:', torchvision.__version__)"

```

A positve output would be:

```bash
Torch version: 2.9.0a0+gita714437
Torchvision version: 0.20.1a0+3ac97aa
```


### Train a workload

Inside the workloads directory you will find 2 models, Vgg19 and GoogleNet. Vgg19 is heavier than GooglNet, for embedded and low performance devices, or faster runs, it is recommended to use GoogleNet.

Both directories contain a python script called `load_pretrained_model.py` which downloads and saves a pretrained model (vgg19 or googlenet). This step is optional (but recommended), as you can upload your own custom pretrained model or weights. For a quick test, this script comes handful.

Also on the workloads directory, you will see a data directory that contains a tiny subset of the ImageNet\_1K dataset. As you can upload your own dataset, using this one is optional, but recommended for quick tests or low performance devices. If using your own dataset, update `train_config.py` (in vgg19) or `config.py` (in googlenet) with the new data path.

Steps guide summary:

- optional and if not previously sourced
```bash
source ~/<your-venv>/bin/activate
```

- optional (you can upload your own weights)
```bash
python load_pretrained_<model>.py 
```

- For a quick training test
```bash
python train.py
```

- To benchmark your machine on a multicore environment
```bash
perf stat -d -d -d python train.py
```

- To benchmark your machine on a single core environment
```bash
taskset 0x1 perf stat -d -d -d python train.py
```
