# biggan-pytorch

Pytorch implementation of LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS (BigGAN)

## train imagenet

for 128\*128\*3 resolution

    python main.py --batch_size 64  --dataset imagenet --adv_loss hinge --version biggan_imagenet --image_path /data/datasets

    python main.py --batch_size 64  --dataset lsun --adv_loss hinge --version biggan_lsun --image_path /data1/datasets/lsun/lsun

    python main.py --batch_size 64  --dataset lsun --adv_loss hinge --version biggan_lsun --parallel True --gpus 0,1,2,3

## Different

* not use cross-replica BatchNorm (Ioffe & Szegedy, 2015) in G

## Compatability

* CPU 
* GPU





