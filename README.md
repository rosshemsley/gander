# 🙎‍♂️ gander
_Let's try playing with GANs_

This package contains a toy implementation of WGAN-gp applied to the CelebA dataset using pytorch-lightning.
We also implement the resolution doubling algorithm proposed by Karras et al. in their _Progressive Growing of Gans_ paper.

This implementation is just a toy, feel free to use it for inspiration, but beware that it is not complete, yet.

## Install the package

Using Python 3.8.1, and a recent pip.
```
pip install git+https://github.com/rosshemsley/gander
```

## Train the net

From within the virtualenv - this is strongly recommended.
```
$ train --root-dir /path/to/dir/containing/celba/dataset
```
