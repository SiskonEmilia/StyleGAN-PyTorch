# StyleGAN-PyTorch

## Usage

Not available yet.

## Checkpoints (pre-trained models)

Not available yet.

## Train on Multiple GPUs



## Changelog

- Working: Umi Iteration
  - PLAN: Support evaluate-only mode
  - PLAN: Support common metrics
  - 5/22: DEBUG: Now this model is able to train on multiple GPUs.
  - 5/22: DEBUG: Fix the bug that the adaptive normalization module does not participate in calculation and back-propagation.
- 5/16 - 5/19: Shiroha Iteration
  - 5/19: Construct [a new anime face dataset](https://github.com/SiskonEmilia/Anime-Wifu-Dataset)
  - 5/16: Able to continue training from a historic checkpoint
  - 5/16: Introduce style-mixing feature
  - 5/16: Debug: Fix the bug that the full connected mapping layer does not participate in calculation and back-propagation.
- 5/13 - 5/15: Kamome Iteration
  - 5/15: Debug: VRAM leak and shared memory conflict
  - 5/14: Debug: Parallel conflict on Windows (Due to the speed limit, we migrate to Linux platform)
  - 5/13: Introduce complete Progressive GAN[2] and its training method.
  - 5/12: Introduce R1 regularization and constant input vector.
  - 5/12: Early implementation of Style-based GAN.

## References

[1] Mescheder, L., Geiger, A., & Nowozin, S. (2018). Which Training Methods for GANs do actually Converge? Retrieved from http://arxiv.org/abs/1801.04406

[2] Karras, T., Aila, T., Laine, S., & Lehtinen, J. (2017). Progressive Growing of GANs for Improved Quality, Stability, and Variation. 1–26. Retrieved from http://arxiv.org/abs/1710.10196