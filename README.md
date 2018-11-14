# biggan-pytorch

Pytorch implementation of LARGE SCALE GAN TRAINING FOR HIGH FIDELITY NATURAL IMAGE SYNTHESIS (BigGAN)

## train imagenet

for 128\*128\*3 resolution

    python main.py --batch_size 64  --dataset imagenet --adv_loss hinge --version biggan_imagenet --image_path /data/datasets

## Different

* not use cross-replica BatchNorm (Ioffe & Szegedy, 2015) in G

## Compatability

* CPU 
* GPU





