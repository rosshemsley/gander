# 🙎‍♂️ gander
_Let's try playing with GANs - a WGAN-gp with a focus on simplicity_

This package contains a toy implementation of WGAN-gp applied to the CelebA dataset using pytorch-lightning.
We also implement the resolution doubling algorithm proposed by Karras et al. in their _Progressive Growing of Gans_ paper.

This implementation is just a toy, feel free to use it for inspiration.

![visualization of training the network](example/training.gif)

_Above: a gif showing the output of the network at as training progresses_

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

## 🌄 Generate a gif from training

The demo gif included in this readme was generated using a bundled CLI tool as follows

From within a virtualenv with gander installed, run
```
$ gengif /path/to/tensorflow/eventlog/events.out.tfevents.xxxxx
```
