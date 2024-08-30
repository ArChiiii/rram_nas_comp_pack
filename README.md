
# rram_nas_comp_pack

The paper titled "RNC: Efficient RAM-aware NAS and Compilation for DNNs on Resource-Constrained Edge Devices" presents a novel framework aimed at optimizing deep neural networks (DNNs) specifically for deployment on edge devices with limited computational resources. 

![hustlin_erd](docs/rnc_overview.png)

## Requirements

install Python (== 3.9) dependencies.
Note: you need to use specific version of the library. See requirements.txt
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n rnc python=3.9
conda activate rnc
pip install -r requirments.txt
```
This project requires PyTorch and torchvision. However, the specific versions you need depend on your CUDA installation. Please install the appropriate versions for your system.

## Options

The following options can be passed as command-line arguments:

### General Options
- `--exp-id`: Experiment ID (default: 'dev')
- `--log-pack`: Enable logging of packing (default: False)
- `--verbose`: Enable verbose output (default: False)

### Model and Packing Options
- `--model`: Model for packing (options: resnet, mbv3_small, squeezenet; default: "mbv3_small")
- `--num-crossbars`: Number of crossbars (default: 10)
- `--depth-split-factor`: Factor for depthwise layer split (default: 1)
- `--layer-threshold`: Threshold for layer conflict in crossbar (default: 1)
- `--xbar-max-load`: Max number of layers in crossbar (default: 6)
- `--no-pack-heuristic`: Disable packing heuristic (default: False)
- `--find-solution`: Enable solution finding with minimun number of crossbar (default: False)

### Optimization Options
- `--margin-factor`: Margin factor for packing
- `--duplicate`: Enable duplication (default: False)

### Evaluation Options
- `--num-sample`: Number of samples to evaluate in cycle/latency evaluation (default: 8)

### Evolution Options
- `--coef-acc`: Weight of accuracy in fitness function (default: 1)
- `--coef-latency`: Weight of latency in fitness function (default: 1)
- `--evolute-method`: Evolution method (options: ndga, ga; default: "ndga")
- `--resume`: Resume evolution from checkpoint (default: False)

### Hardware Options
- `--no-cuda`: Forbid CUDA usage (default: False)
- `--device`: Which GPU to use (default: 0)
- `--seed`: Random seed (default: 123)


## Compilation with packing








## RRAM-NAS 







## Acknowledgements
This project builds upon the 

- [Once for All](https://github.com/mit-han-lab/once-for-all) 
- [MNSIM-2.0](https://github.com/thu-nics/MNSIM-2.0)


We thank the authors for their valuable work.