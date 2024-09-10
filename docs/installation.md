# Installing dependencies for DreamHOI

### Prerequisites
To run with the full DeepFloyd IF + MVDream supervision, an NVIDIA GPU with at least 30GB VRAM is usually needed. The exact GPU memory needed also depends on a case-by-case basis. You can disable MVDream (which should now fit training into 24GB), and/or lower the rendering resolution and/or batch size (see configuration options in [README.md](README.md)).

The code is tested on Debian, though any Linux environment should work.

### Environment setup
Clone the repository:
```sh
git clone https://github.com/hanwenzhu/dreamhoi.git
git submodule update --init --recursive
cd dreamhoi
```

The code is tested on the following versions, and we highly recommend following the versions exactly (CUDA 11.3, GCC 9.5, PyTorch 1.12.1). We recommend using a new conda environment:
```sh
conda create -n cuda113 -y python=3.9 cudatoolkit=11.3 cudatoolkit-dev=11.3 gcc_linux-64=9.5.0 gxx_linux-64=9.5.0
conda activate cuda113
```
Inside the conda environment, create a virtual environment for PyTorch 1.12.1:
```sh
pip install virtualenv
python3 -m virtualenv venv
. venv/bin/activate
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 pytorch-lightning --extra-index-url https://download.pytorch.org/whl/cu113
```

Every time you run our code, you should first run:
```sh
conda activate cuda113
. venv/bin/activate
export LIBRARY_PATH=$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
```
to properly activate the environment.

### Install dependencies for MVDream and threestudio
Inside the venv, install tiny-cuda-nn (ensure you are in a GPU environment during installation):
```sh
export LIBRARY_PATH=$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
pip install -vvvvv "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```
Install the rest of the threestudio requirements:
```sh
# inside dreamhoi
pip install -r src/MVDream-threestudio/requirements.txt
```
Downgrade bitsandbytes so that it works with CUDA 11.3:
```sh
pip install bitsandbytes==0.38.1
```

Install embree (for intersection regularizer):
```sh
conda install pyembree==0.1.6
```
and edit `venv/pyvenv.cfg` to set
```cfg
include-system-site-packages = true
```
so that `pyembree` is visible inside the virtual environment.

Install MVDream:
```sh
# inside dreamhoi
cd src/MVDream-threestudio
git submodule update --init --recursive
pip install -e extern/MVDream
```

For huggingface to access DeepFloyd IF, you need to accept the [license](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) with your account, and run
```sh
huggingface-cli login
```

Note that the directory `src/MVDream-threestudio/.threestudio_cache/` will be used for caching model weights (so faster read speeds help), which can take 2GB. The directory `src/MVDream-threestudio/outputs/` will store intermediate NeRF results which can take some space, and `smplify/` will be used to store results.

### Download SMPL models
Parts of our code uses [SMPL](https://smpl.is.tue.mpg.de) and its derivatives. Our paper presents results on the SMPL+H models, although you may also use plain SMPL models for our pipeline. We download these models (following their [instructions](https://github.com/vchoutas/smplx/blob/main/README.md)). You need to download the VPoser model and either the 3 SMPL models or the 2 SMPL+H models (or both) as below.

##### Download SMPL models
From the “Downloads” page of [SMPL](https://smpl.is.tue.mpg.de) download “version 1.0.0 for Python 2.7” for male and female models. In order to download, you need to first log in and accept the license. Then download the gender-neutral model at [SMPLify](https://smplify.is.tue.mpg.de/download.php) “Downloads” page, inside `smplify_code_v2.zip`. Again you would need to log in and accept terms. You can use any one of the genders for our pipeline (see README).

Follow [these instructions](https://github.com/vchoutas/smplx/tree/main/tools#removing-chumpy-objects) to prepare the `.pkl` models (it changes chumpy arrays to numpy arrays). The script runs in Python 2 (or you can edit manually to make it run in Python 3). Renaming the `.pkl` files appropriately, in the end there should be `SMPL_FEMALE.pkl`, `SMPL_MALE.pkl`, and `SMPL_NEUTRAL.pkl`.

##### Download SMPL+H models
Go to [MANO](https://mano.is.tue.mpg.de) and log in and accept the terms for download. Go to “Downloads” page, then click “Models & Code” to download a zip file with 4 `.pkl` files inside. Follow [these instructions](https://github.com/vchoutas/smplx/tree/main/tools#removing-chumpy-objects) to prepare the `.pkl` models (it changes chumpy arrays to numpy arrays). The script runs in Python 2 (or you can edit manually to make it run in Python 3). Then follow [these instructions](https://github.com/vchoutas/smplx/blob/main/tools/README.md#merging-smpl-h-and-mano-parameters) to merge the 4 files into 2, one for each gender: `SMPLH_FEMALE.pkl`, `SMPLH_MALE.pkl`.

##### Download VPoser (required)
Go to [SMPL-X](https://smpl-x.is.tue.mpg.de), log in and accept terms, and download “VPoser v1.0” in the “Downloads” page. You should find `snapshots/TR00_E096.pt` in the downloaded zip.

##### Place models in dreamhoi
Please put the above prepared models into appropriate places, inside `dreamhoi/src/MultiviewSMPLifyX`:
```
dreamhoi/src/MultiviewSMPLifyX
├── smplx
│   └── models
│       ├── smpl
│       │   ├── SMPL_FEMALE.pkl
│       │   ├── SMPL_MALE.pkl
│       │   └── SMPL_NEUTRAL.pkl
│       └── smplh
│           ├── SMPLH_FEMALE.pkl
│           └── SMPLH_MALE.pkl
└── vposer
    └── models
        └── snapshots
            └── TR00_E096.pt
```

### Set up SMPLify
To satisfy requirements of SMPLify-X, run inside the venv
```sh
pip install configargparse configer torchgeometry==0.1.2
```
`torchgeometry` has a bug in PyTorch 1.12.1, and to fix it you need to edit `venv/lib/python3.9/site-packages/torchgeometry/core/conversions.py` and change all 4 instances of `(1 - mask_…)` to `(~mask_…)` in lines 302–304 of this file.

### Install OpenPose
Follow OpenPose's [instructions](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md) to install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) at some location on your system. Make sure that all models, including the face and hands models are downloaded (see [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/1_prerequisites.md#general-tips)). Note that the venv and conda environment may need to be deactivated during installation.

*OpenPose is difficult to install on headless servers without sudo rights, so it may need different environment flags during run-time, or it may need to be installed on a separate system or Docker. Depending on your case, you may want to modify `run_openpose` in `dreamhoi/main.py` to reflect your settings.*

## Questions
Please raise any issues encountered to the respective dependencies or [to us](https://github.com/hanwenzhu/dreamhoi/issues/new).
