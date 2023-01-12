
# AE Result
- noise image in training stage

<img src="saved/train/noise_train_0003_0000.jpg"><p>
- denoise image (auto-encoder) in training stage

<img src="saved/train/denoise_train_0003_0000.jpg"><p>
- noise image in validation stage

<img src="saved/val/noise_val_0002_0000.jpg"><p>
- denoise image (auto-encoder) in valiation stage<p>

<img src="saved/val/denoise_val_0002_0000.jpg">

# VAE Result
- train stage: 
  - raw image \
<img src="saved_vae/train/raw_train_0004_0000.jpg"><p>
  - generated_img \
<img src="saved_vae/train/gen_train_0004_0000.jpg"><p>

- val stage:
  - raw image \
<img src="saved_vae/val/raw_val_0007_0000.jpg"><p>
  - generated_img \
<img src="saved_vae/val/gen_val_0007_0000.jpg"><p>


# Usage
## install package

```shell
conda create -n auto python=3.6
conda activate auto
cd ./auto-encoder

pip install -r ./requirements.txt

python train.py  # for AE
python vae_train.py # for VAE
```

# Reference 

## VAE
- https://adaning.github.io/posts/9047.html
- https://blog.csdn.net/heyc861221/article/details/80130968
- https://www.cnblogs.com/weilonghu/p/12567793.html
- https://zhuanlan.zhihu.com/p/144649293 (very good article!!!)

### github refer
- https://github.com/AntixK/PyTorch-VAE
- https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py

## Binary cross entropy
- https://blog.csdn.net/Cy_coding/article/details/116427968

