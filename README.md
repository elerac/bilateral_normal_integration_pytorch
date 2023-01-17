# Bilateral Normal Integration: PyTorch

Unofficial implementation of [**Bilateral Normal Integration** (ECCV2022)](https://github.com/hoshino042/bilateral_normal_integration) with PyTorch. The official implementation solves the optimization problem (19) with IRLS by rewriting the problem into matrix form (Eq. (20)). Instead of their approach, I tackled solving it by using the backpropagation algorithm in PyTorch.

In conclusion, PyTorch successfully reproduces the same results as the official implementation. However, it is unstable and requires much computation time. So the official implementation is preferred for practical use.

Nevertheless, I believe this PyTorch implementation is beneficial to understanding because I write the calculation process explicitly instead of in complex matrix form. Moreover, it can be easily integrated with additional constraints thanks to automatic differentiation in PyTorch.

## License

This repository contains codes and datasets from [hoshino042/bilateral_normal_integration](https://github.com/hoshino042/bilateral_normal_integration). The original repository is licensed under the GPL-3.0 license, and the same license applies to the materials used in this repository.
