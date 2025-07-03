# AFDiT: flow-guided transformer diffusion for structure-aware virtual try-on

## Abstract

> Virtual try-on aims to generate photorealistic images of a person wearing a target garment, requiring precise garment–body alignment, fine-grained texture preservation, and robustness to pose variations. We propose AFDiT, a structure-aware virtual try-on framework that integrates appearance flow estimation with transformer-based diffusion generation. Specifically, we design an appearance flow encoder–decoder to predict dense garment deformation and alignment, and a parsing-guided mask fusion strategy to refine inpainting regions while preserving uncovered body parts. For high-quality synthesis, we introduce a warped garment-guided stable diffusion pipeline, injecting garment-specific CLIP embeddings and flow-guided warped garment features into a transformer-based diffusion model. Extensive experiments on VITON-HD demonstrate that AFDiT outperforms state-of-the-art methods in both paired and unpaired settings, achieving superior alignment, realism, and generalization to diverse poses and garments.

## Environment

```shell
conda create -n afdit python=3.12 -y
conda activate afdit
pip install -r requirements.txt  # for CUDA 12.4
```

## Training

Modify your parameters in the shell script files (*.sh), e.g., train_aflow_enc_dec.sh and train_viton.sh, to adjust hyperparameters or dataset paths.

```shell
bash train_aflow_enc_dec.sh
bash train_viton.sh
```

## Acknowledgement

Our code is based on the implementation of [FitDiT: Advancing the Authentic Garment Details for High-fidelity Virtual Try-on](https://github.com/BoyuanJiang/FitDiT) [[Paper](https://arxiv.org/abs/2411.10499)].

## Citation

Our work is now available [online](https://link.springer.com/article/10.1007/s00371-025-04075-5). If you find our work helpful for your research, please consider citing it.
```
@article{Ding2025afdit,
  author    = {Huiming Ding and Beining Wu and Mengtian Li and Zhifeng Xie},
  title     = {AFDiT: flow-guided transformer diffusion for structure-aware virtual try-on},
  journal   = {The Visual Computer},
  year      = {2025},
  month     = {jun},
  day       = {29},
  eissn     = {1432-2315},
  issn      = {0178-2789},
  doi       = {10.1007/s00371-025-04075-5},
  url       = {https://doi.org/10.1007/s00371-025-04075-5},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MTKSHU/AFDiT&type=Date)](https://www.star-history.com/#MTKSHU/AFDiT&Date)
