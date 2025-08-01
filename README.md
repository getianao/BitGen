
# Interleaved Bitstream Execution for Multi-Pattern Regex Matching on GPUs

This is the repository for MICRO'25 paper:
 "Interleaved Bitstream Execution for Multi-Pattern Regex Matching on GPUs".

BitGen is a compiler that generates optimized CUDA kernels from bitstream programs for high-performance regular expression matching.


For example, the regex `(ab)|c` is compiled into the following bitstream operation: $S = (((S_a \gg 1)$ & $S_b) \gg 1)$ | $S_c$, where $S_a$, $S_b$, and $S_c$ are bitstreams indicating the presence of each character class at every position over the input string. BitGen parallelizes the execution of bitstream operations on the GPU.

## 0. Requirements

- Hardware:
    ```
    - CPU x86_64 with >= 32 GB RAM
    - NVIDIA GPU with compute capability >= sm_86 and >= 24 GB device memory
    ```
    We have tested our project on an NVIDIA RTX 3090 (Ampere architecture, 24 GB memory), an NVIDIA H100 (Hopper architecture, 94 GB memory), 
    and an NVIDIA L40S (Ada Lovelace architecture, 48 GB memory).

- OS & Software:

    ``` bash
    - Ubuntu 20.04
    - GCC >= 13
    - CUDA >= 12.4 and NVCC >= 12.4
    - Python >= 3.10
    ```

## 1. Environment Setup

### 1.1 Clone the Repository and Download Benchmark

```bash
git clone --recursive git@github.com:getianao/BitGen.git
cd BitGen && source env.sh && echo ${BITGEN_ROOT}     # set environment variables
# Download benchmarks (~1.7 GB)
wget https://hkustgz-my.sharepoint.com/:u:/g/personal/tge601_connect_hkust-gz_edu_cn/ES7vHG6o711Pp9Bpj2tr5hEB-RLa_ygdGbYIjxY6MT4spQ?e=iL6Hef\?e\=5bWc4W\&download=1 -O datasets_bitstream.tar.gz
mkdir -p datasets && tar -xzvf datasets_bitstream.tar.gz -C datasets
```

### 1.2 Install Dependencies

We recommend to use Docker to setup the environment. We provide a [dockerfile](docker/Dockerfile) in the docker folder. 
You can also setup the environment manually.


#### 1.2.1 Docker (Recommended)

If you don't have Docker installed, please follow the [NVIDIA Container Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) to install Docker using the following commands:

```bash
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Next, build the Docker image and launch a container. This process will take approximately 30 minutes.
```bash
# Build the Docker image
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t bitgen-ae ./docker
# Launch the Docker container
docker run -it --rm --gpus all -v $(pwd):/BitGen bitgen-ae:latest /bin/bash
```
You will automatically be placed inside the container's shell, where the environment is fully configured.


#### 1.2.2 Manual Setup

If you prefer a manual setup, install the required dependencies as follows.
```bash
## BitGen
# sudo add-apt-repository ppa:ubuntu-toolchain-r/test && sudo apt-get update
sudo apt-get install -y gcc-13 g++-13
sudo apt-get install libgraphviz-dev libboost-all-dev time
conda create --name bitgen-py310 python=3.10 -y
conda activate bitgen-py310
conda config --add channels conda-forge
conda install cuda-cccl cuda-version=12.4 -y
pip install typing-extensions --index-url https://pypi.org/simple
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install cuda-python==12.8 numpy scipy matplotlib networkx colorama pandas bitarray tqdm torchviz pynvml colorlog pyyaml pygraphviz
pip install git+https://github.com/getianao/figurePlotter.git@2c65e40a8f017fbde058a6fe09b315ee64da4301
## ngAP
sudo apt-get install -y libtbb-dev=2020.1-2 cmake
## Hyperscan
sudo apt-get install -y ragel libboost-all-dev nasm libsqlite3-dev pkg-config 
# sudo echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial main" |  tee -a /etc/apt/sources.list
# sudo echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe" |  tee -a /etc/apt/sources.list
# sudo apt-get update 
sudo apt-get install -y g++-5 gcc-5
```

Proceed to the next steps within the configured environment (either the Docker container or your manual setup).



## 2. Building the Artifact

BitGen translates bitstream programs into CUDA code. In this artifact, we use `icgrep` to convert regular expressions into bitstream programs, which serve as input to BitGen.

```bash
# Building BitGen
pip install .
# Building Baseline
./scripts/install_icgrep.sh
./scripts/install_hs.sh
./scripts/install_ngap.sh # manually update -arch in ngAP/code/CMakeLists.txt if needed
```

## 3. Quick Start: Running a Small Example

To verify that the setup is correct, run BitGen on a small dataset. 
The following command compiles 8 regexes extracted from Snort, executes the generated CUDA code on the GPU, and validates the results against `icgrep`.

``` bash
LOG=DEBUG python ${BITGEN_ROOT}/scripts/run_bitgen.py \
    -f=${BITGEN_ROOT}/datasets_small/snort_small.regex \
    -i=${BITGEN_ROOT}/datasets_small/snort_10KB.input \
    --input-size=10000 --repeat-input=1 --multi-input=1  --split-input=1 --regex_num_from_file=-1 --regex_group_number=4 \
    --parallel_compile_parabix=1 --backend=cuda --parallel_compile_cuda=1 --parallel_compile_cuda_lto=0 \
    --check 1 \
    --pass-graph-break=-1 \
    --pass-cc-advanced=1 --pass-cc-advanced-max=8 \
    --pass-short-circuit=1 --pass-short-circuit-start=2 --pass-short-circuit-interval=1 --pass-short-circuit-syncpoint=1
```

If the execution is successful, you will see timing data and a validation message.
```
...
Validation Passed: count 16
----- Timing Data -----
  group   exe   app       name  exec_num  duration  avg_duration  input_size  throughput  count  ref_count check
0  None  None  None  transpose         8  0.111982      0.013998    0.009552  698.770861    NaN        NaN   NaN
1  None  None  None  run_regex         8  0.140015      0.017502    0.009552  558.870096   16.0       16.0  True
Total count: 16.0
Total count ref: 16.0
True count: 1
False count: 0
-----------------------
```

Generated code is stored in the `./generated_code` directory.

**Note on Match Counts**:  
The match count reported by BitGen may occasionally differ from the reference count.  
One reason is that BitGen does not deduplicate matches from overlapping patterns (e.g., `ab` and `a*b` matching "ab") when they are assigned to different regex groups. In contrast, the baseline may report only a single match in such cases.


The following flags control our proposed optimizations. Use `--help` for more details.


| Optimization Option         | Description                                                                                     |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| `--pass-graph-break`        | Dependency-Aware Thread-Data Mapping. Controls loop fusion. -1: Static+Dynamic , 0: Static only, 1: No fusion, 2: Simple fusion.            |
| `--pass-cc-advanced`        | Shift Rebalancing. 0: Disable. 1: Enable.                                           |
| `--pass-cc-advanced-max`    | Merge Size for shift Rebalancing.            |
| `--pass-short-circuit`      | Zero Block Skipping. 0: Disable. 1: Enable.              |                                 |
| `--pass-short-circuit-interval` | Interval size for Zero Block Skipping.                              |



## 4. Reproducing Paper Results

The following scripts will run the evaluation to reproduce the main results from the paper.

- Figure 11, Table 2: Throughput results for BitGen and baselines.
- Figure 12 : Performance breakdown of BitGen's optimizations.

### 4.1. Run Experiments

These scripts execute the throughput and breakdown experiments. The process may take several hours.
```bash
./scripts/run_throughput.sh
./scripts/run_breakdown.sh
```

**Note**: The provided configurations in the `config/` directory are tuned for an NVIDIA RTX 3090. 
Performance results may vary on different CPUs and GPUs. 
You can adjust parameters in the `exec_*.yaml` files or add new application configurations in `app_*.yaml`.


All resulting data will be saved as CSV files in the `results/csv` and `raw_results/ac` directory, 
and log files will be placed in the `log` folder with filenames based on timestamps.

### 4.2 Plot Results

Use the following scripts to process the raw data and generate the figures.
```bash
# Figure 11, Table 2
python ./scripts/plot/plot_app_full_new.py
# Figure 12
python ./scripts/plot/plot_app_full_breakdown.py 
```
The resulting plots will be saved in the `results/figures` and `results/tables` directory.

For your reference, we have included results collected on the NVIDIA RTX 3090 and Intel Xeon Silver 4214R CPU, aw well as the figures and tables in the `ref_result` folder.

## Paper
Please refer to this paper for more details.

```bibtex 
@inproceedings{micro25bitgen, 
  title={Interleaved Bitstream Execution for Multi-Pattern Regex Matching on GPUs}, 
  author={Tianao Ge, Xiaowen Chu, and Hongyuan Liu}, 
  booktitle={Proceedings of The 58th IEEE/ACM International Symposium on Microarchitecture (MICRO 2025)}, 
  year={2025} 
} 
```

