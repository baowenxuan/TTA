# Datasets

## Corruption Datasets

Please download the data from the [robustness benchmark](https://github.com/hendrycks/robustness). It provides these
datasets:

- [CIFAR-10-C](https://zenodo.org/record/2535967)
- [CIFAR-100-C](https://zenodo.org/record/3555552)
- [ImageNet-C](https://zenodo.org/record/2235448)
- [Tiny-ImageNet-C](https://zenodo.org/record/2536630)

The directory structure should look like

```
${data_root}/corruption
│ 
├── CIFAR-10-C
│   ├── brightness.npy
│   ├── ...
│   ├── pixelate.npy
│   └── labels.npy
│ 
├── CIFAR-100-C
│   ├── brightness.npy
│   ├── ...
│   ├── pixelate.npy
│   └── labels.npy
│ 
├── ImageNet-C
│   ├── brightness
│   │   ├── 1
│   │   ├── ...
│   │   └── 5
│   │       ├── n01440764
│   │       │   ├── ILSVRC2012_val_00000293.JPEG
│   │       │   ├── ...
│   │       │   └── ILSVRC2012_val_00048969.JPEG
│   │       ├── ...
│   │       └── n15075141
│   ├── ...
│   └── pixelate
│
└── Tiny-ImageNet-C
    ├── brightness
    │   ├── 1
    │   ├── ...
    │   └── 5
    │       ├── n01443537
    │       │   ├── test_1103.JPEG
    │       │   ├── ...
    │       │   └── test_9640.JPEG
    │       ├── ...
    │       └── n12267677
    ├── ...
    └── pixelate
```

We also provide the clean version of these corrupted images

- [CIFAR-10](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html)
- [CIFAR-100](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CIFAR100.html)
- [ImageNet](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html)
- [Tiny-ImageNet](https://drive.google.com/file/d/1pTkzhONG8o2Zx9ocYqjmqY3G5ejtzFjr/view?usp=sharing)

PyTorch can automatically download CIFAR-10 and CIFAR-100 for you, while ImageNet and Tiny-ImageNet need to be
downlaoded manually.

The directory structure should look like

```
${data_root}/torchvision
│ 
├── cifar-10-batches-py
│ 
├── cifar-100-python
│ 
├── ImageNet
│   └── val
│       ├── n01440764
│       │   ├── ILSVRC2012_val_00000293.JPEG
│       │   ├── ...
│       │   └── ILSVRC2012_val_00048969.JPEG
│       ├── ...
│       └── n15075141
│
└── Tiny-ImageNet
    └── imagenet_val_bbox_crop
        ├── n01443537
        │   ├── test_1103.JPEG
        │   ├── ...
        │   └── test_9640.JPEG
        ├── ...
        └── n12267677
```

A common practice is to downsample 5k imagenets from the 50k images in ImageNet validation set. To use this subset, we
provide `ImageNetC5K` and `ImageNet5K`, respectively. 

## ImageNet Variants (a.k.a. Natural Distribution Shift)

- ImageNet-A
- ImageNet-V2
- ImageNet-R
- ImageNet-Sketch

Prepare the data according to [this link](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md). 

## DomainBed Datasets

Please download the data according to the instructions in [DomainBed](https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py). 