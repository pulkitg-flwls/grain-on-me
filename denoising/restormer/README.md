# Restormer: Efficient Transformer for High-Resolution Image Restoration

Codebase performing denoising of images using [Restormer](https://github.com/swz30/Restormer).

## Setup

1. Make conda environment
```
conda create -n pytorch181 python=3.7
conda activate pytorch181
```

2. Install dependencies
```
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```

3. Install basicsr
```
python setup.py develop --no_cuda_ext
```

4. Download pretrained checkpoints

Download the pre-trained [model](https://drive.google.com/file/d/1FF_4NTboTWQ7sHCq4xhyLZsSl0U0JfjH/view?usp=sharing) and place it in `./pretrained_models/`

## Inference
```
python demo.py --input_dir <input_dir>/ --result_dir <output_dir> 
```
Optional argument `--tile` if image is OOM. Works with 1024 for 2k/4k resolution images
