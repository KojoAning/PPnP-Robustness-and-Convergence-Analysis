# PyTorch Implementation of P²nP Paper

**Reference:**  
[Tao Hong](https://hongtao-argmin.github.io), Xiaojian Xu, Jason Hu, and [Jeffrey A. Fessler](https://web.eecs.umich.edu/~fessler/), [Provable Preconditioned Plug-and-Play Approach for Compressed Sensing MRI Reconstruction](https://arxiv.org/abs/2405.03854), IEEE Transactions on Computational Imaging, vol. 10, pp. 1476–1488, Oct. 2024.

---

## About This Repository

This repository extends the original P²nP implementation to support a broader experimental evaluation across two trajectory types and two anatomies:

- **Spiral trajectory** (multi-interleave) evaluated on brain MRI (MoDL dataset, test index 60)
- **Golden-angle radial trajectory** evaluated on knee MRI (fastMRI dataset, slice 12)

Four methods are compared across multiple acceleration factors R:

| Method | Description |
|---|---|
| **PnP-ISTA** | Standard PnP with no preconditioner (baseline) |
| **P²nP-F-1** | Fixed Neumann polynomial preconditioner (degree 1) |
| **P²nP-F-Cheb** | Fixed Chebyshev polynomial preconditioner (degree 1) |
| **P²nP-D** | Dynamic quasi-Newton preconditioner (ZMSHR1) |

---

## Getting Started

### Installation

1. Create and activate the conda environment:
   ```bash
   conda env create -f PPnP.yml
   conda activate unity
   ```
   Or install dependencies directly:
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** PyTorch is listed with a CUDA 12.6 suffix (`+cu126`). For a different CUDA version or CPU-only setup, install PyTorch separately from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above.

2. Download the trained DRUNet denoiser weights from [Google Drive](https://drive.google.com/drive/folders/1QN7t0l3PnzHqvx9d-npONiF8Sz946Oi_?usp=share_link) and place them in the project root:
   - `norm-equivMSECh1_20240320_SE_DRUCNNBrainDenoise_sigma0.1.pth`
   - `norm-equivMSECh1_20240320_SE_DRUCNNKneeDenoise_sigma0.1.pth`

3. The coil sensitivity maps and spiral trajectory files are included in the repository under `data/spiral/`:
   - `trj.npy` – full spiral trajectory (32 arms, 422 KB)
   - `mpsSim32.npy` – simulated 32-coil sensitivity maps (16 MB)

### Data

- **Brain MRI:** MoDL dataset (`data/dataset.hdf5`) — download from [DATASET_URL]
- **Knee MRI:** fastMRI dataset (`data/file1000000.h5`) — download from [DATASET_URL]

---

## Files

| File | Description |
|---|---|
| `ppnp.ipynb` | Main notebook: data loading, experiment loops, plots, and reconstruction grids |
| `demoReco.py` | Original demo from Hong et al. for a quick single-run example |
| `optalg_Tao_Pytorch.py` | Optimization algorithms: `ISTA_PnP`, `DynamicPnP`, `Power_Iter`, etc. |
| `optpoly.py` | Chebyshev polynomial coefficient utilities (requires the [Chebyshev](https://github.com/mlazaric/Chebyshev) package) |
| `dataset.py` | Dataset loaders for `BrainMRIDataset` (MoDL) and `FastMRIKneeDataset` (fastMRI) |
| `requirements.txt` | Python dependencies (versions from the `unity` conda environment) |
| `results/` | Output folder: CSV result tables, PSNR plots, GIF animations, reconstruction grid PNGs |

---

## Running the Experiments

Open and run `ppnp.ipynb` in order. The notebook is organised into the following sections:

1. **Setup** – load models, images, and NUFFT operators
2. **Spiral experiment** – vary spiral arm count across 5 acceleration levels (R ≈ 50× to 402×)
3. **Radial experiment** – vary spoke count across 6 acceleration levels (R ≈ 8× to 128×)
4. **Plots** – PSNR vs iteration, PSNR vs runtime, SSIM vs R, fixed-point residuals
5. **Reconstruction grids** – visual comparison of all four methods with error maps and zoomed ROI
6. **CSV export** – save results to `results/Spiral_results.csv` and `results/Radial_results.csv`
7. **Animations** – animated PSNR vs R and PSNR vs time GIFs saved to `results/`
8. **Paper figures** – static PNG figures for the written report

---

## Results Summary

### Spiral (Brain MRI)

- All three preconditioned methods outperform PnP-ISTA at every acceleration level.
- P²nP-D achieves the highest PSNR at low-to-moderate R (up to +5.02 dB over PnP-ISTA at R ≈ 50×).
- P²nP-F-Cheb leads at extreme undersampling (R ≈ 402×), where noisy gradients degrade P²nP-D.

### Radial (Knee MRI)

- Preconditioner gains are smaller than for spiral, reflecting the less uniform spectral structure of radial sampling.
- P²nP-D degrades at high R (≥ 32×) due to noise-dominated gradient estimates.
- P²nP-F-Cheb is the most robust method, leading from R ≈ 16× onward.

---

## Code Attribution

The experimental code is based on the original implementation released by Hong et al. as part of the paper cited above. It was adapted and extended to support additional trajectory configurations (radial), cross-dataset evaluation (fastMRI knee), and the analysis presented in the accompanying report.

---

## Contact

For questions about the original P²nP method, contact **tao.hong@austin.utexas.edu**.
