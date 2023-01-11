
# Result
- noise image in training stage

<img src="saved/train/noise_train_0003_0000.jpg"><p>
- denoise image (auto-encoder) in training stage

<img src="saved/train/denoise_train_0003_0000.jpg"><p>
- noise image in validation stage

<img src="saved/val/noise_val_0002_0000.jpg"><p>
- denoise image (auto-encoder) in valiation stage<p>

<img src="saved/val/denoise_val_0002_0000.jpg">

# Usage
## install package

```shell
conda create -n auto python=3.6
conda activate auto
cd ./auto-encoder

pip install -r ./requirements.txt

python train.py
```