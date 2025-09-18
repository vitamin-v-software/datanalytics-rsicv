# VGG19 – PyTorch (Modified)

This directory is a **modified version** of [VGG-PyTorch](https://github.com/ZhiqiangZhangCUG/VGG-PyTorch),  

## Additions in This Fork

- `load_pretrained_vgg11.py`: downloads and saves a pretrained model to `results/pretrained_models/`.
- `train_distributed.py`: mechanism to train the model across multiple nodes.
- `train_distributed_debug.py`: distributed training with detailed debug statements.

All other scripts (training, testing, dataset preparation) remain compatible with the original repository.

## References

- Original code: [VGG-PyTorch](https://github.com/ZhiqiangZhangCUG/VGG-PyTorch)
- Paper: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556v6.pdf)

## License

This repository retains the license of the original work (see [LICENSE](LICENSE)).
