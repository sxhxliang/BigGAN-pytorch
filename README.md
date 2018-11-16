# biggan-pytorch

Pytorch implementation of LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS (BigGAN)

## train imagenet

for 128\*128\*3 resolution

    python main.py --batch_size 64  --dataset imagenet --adv_loss hinge --version biggan_imagenet --image_path /data/datasets

    python main.py --batch_size 64  --dataset lsun --adv_loss hinge --version biggan_lsun --image_path /data1/datasets/lsun/lsun

    python main.py --batch_size 64  --dataset lsun --adv_loss hinge --version biggan_lsun --parallel True --gpus 0,1,2,3 --use_tensorboard True



## Different

* not use cross-replica BatchNorm (Ioffe & Szegedy, 2015) in G

## Compatability

* CPU 
* GPU

## Results

LSUN DATASETS(two classes): classroom and church_outdoor
* iter 82200 (128x128) batch_size 64
![](./results/iter_82200_fake.png)





