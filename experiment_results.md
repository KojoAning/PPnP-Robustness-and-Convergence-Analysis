# P²nP Full Experimental Setup and Results

**Reference:** Tao Hong et al., *Provable Preconditioned Plug-and-Play Approach for Compressed Sensing MRI Reconstruction*, IEEE TCI, 2024.
**Device:** NVIDIA GPU (cuda:0) | **Iterations:** 100 | **Image:** Brain GT #1 (256×256)

---

## 1. Setup

### 1.1 Imports and Model

```python
import torch
import scipy.io
import matplotlib.pyplot as plt
import torchkbnufft as tkbn
import optalg_Tao_Pytorch as opt
from models.drunet import DRUNet
from skimage.metrics import structural_similarity as ssim_metric
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

im_type    = 'Brain'   # 'Brain' or 'Knee'
model_path = 'norm-equivMSECh1_20240320_SE_DRUCNNBrainDenoise_sigma0.1.pth'

model = DRUNet(blind=True, mode='norm-equiv', in_nc=1, out_nc=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
```

### 1.2 Load and Normalize Ground Truth Image

```python
data         = scipy.io.loadmat('image/BrainDeepGT_1.mat')
im           = data['im_real'] + 1j * data['im_imag']
im           = im / np.max(np.abs(im))   # normalize max magnitude to 1
im_original  = im.copy()
```

### 1.3 Load Trajectory and Sensitivity Maps

```python
noise_level = 1e-3   # brain: 1e-3 | knee: 3e-4

trj_full = np.load('data/spiral/trj.npy')    # shape: (32, 1688, 2)
mps      = np.load('data/spiral/mpsSim32.npy')  # shape: (32, 256, 256)
```

**Output:**
```
trj_full: (32, 1688, 2) | subsampled (paper setting, step=6): (6, 1688, 2)
SNR: 14.54 dB
```

### 1.4 Build NuFFT Operators

```python
im_size     = im.shape
im_torch    = torch.tensor(im).unsqueeze(0).unsqueeze(0).to(torch.complex64).to(device)
nufft_ob    = tkbn.KbNufft(im_size=im_size, device=device)
adjnufft_ob = tkbn.KbNufftAdjoint(im_size=im_size, device=device)
smaps       = torch.tensor(mps).unsqueeze(0).to(device)
```

---

## 2. Helper Functions

### 2.1 Spiral Forward Operator

```python
def build_spiral_ops(step):
    """Normalized spiral forward/adjoint for a given step (arm subsampling)."""
    trj_sub     = trj_full[0:-1:step, :, :]
    n_arms      = len(trj_sub)
    trj_reshape = trj_sub.reshape(1, trj_sub.shape[0] * trj_sub.shape[1], trj_sub.shape[2])
    kt          = torch.tensor(trj_reshape, dtype=torch.float).permute(0, 2, 1).to(device)
    Ax_r   = lambda x, k=kt: nufft_ob(x, k, smaps=smaps.to(x))
    ATx_r  = lambda x, k=kt: adjnufft_ob(x, k, smaps=smaps.to(x))
    L_s    = opt.Power_Iter(Ax_r, ATx_r, im_size, tol=1e-6, device=device)
    L_sr_s = torch.sqrt(L_s)
    Ax_n   = lambda x, k=kt, l=L_sr_s: nufft_ob(x, k, smaps=smaps.to(x)) / l
    ATx_n  = lambda x, k=kt, l=L_sr_s: adjnufft_ob(x, k, smaps=smaps.to(x)) / l
    return Ax_n, ATx_n, n_arms
```

> **Note:** `L` (spectral norm of `AᴴA`) is recomputed per acceleration level because removing arms changes the operator's energy.
> Default argument trick (`k=kt`, `l=L_sr_s`) avoids Python closure issues in loops.

### 2.2 Radial Forward Operator (Golden-Angle)

```python
def build_radial_ops(nspokes, spokelength=1024):
    """Normalized golden-angle radial forward/adjoint for a given number of spokes."""
    ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
    kx = np.zeros((nspokes, spokelength, 1))
    ky = np.zeros((nspokes, spokelength, 1))
    ky[0, :, 0] = np.linspace(-np.pi, np.pi, spokelength)
    for i in range(1, nspokes):
        kx[i,:,0] = np.cos(ga)*kx[i-1,:,0] - np.sin(ga)*ky[i-1,:,0]
        ky[i,:,0] = np.sin(ga)*kx[i-1,:,0] + np.cos(ga)*ky[i-1,:,0]
    trj_r       = np.concatenate((kx, ky), axis=2)
    trj_reshape = trj_r.reshape(1, nspokes * spokelength, 2)
    kt          = torch.tensor(trj_reshape, dtype=torch.float).permute(0, 2, 1).to(device)
    Ax_r   = lambda x, k=kt: nufft_ob(x, k, smaps=smaps.to(x))
    ATx_r  = lambda x, k=kt: adjnufft_ob(x, k, smaps=smaps.to(x))
    L_s    = opt.Power_Iter(Ax_r, ATx_r, im_size, tol=1e-6, device=device)
    L_sr_s = torch.sqrt(L_s)
    Ax_n   = lambda x, k=kt, l=L_sr_s: nufft_ob(x, k, smaps=smaps.to(x)) / l
    ATx_n  = lambda x, k=kt, l=L_sr_s: adjnufft_ob(x, k, smaps=smaps.to(x)) / l
    return Ax_n, ATx_n
```

### 2.3 Simulate Noisy Measurements

```python
def simulate_measurements(Ax_fn, nl):
    """Apply forward model to im_torch and add complex Gaussian noise at level nl."""
    b          = Ax_fn(im_torch)
    bm, bn, bk = b.shape
    torch.manual_seed(2); nr = torch.randn(bm, bn, bk, device=device)
    torch.manual_seed(5); ni = torch.randn(bm, bn, bk, device=device)
    return b + nl * (nr + 1j * ni)
```

### 2.4 Run All Three Reconstruction Methods

```python
def run_all_methods(Ax_fn, ATx_fn, b_n, n_iters):
    """Run PnP-ISTA, Pre1-PnP and Dynamic-PnP; return dict of results."""
    res = {}

    # PnP-ISTA — no preconditioner
    x, psnr, t, fixed = opt.ISTA_PnP(
        n_iters, Ax_fn, ATx_fn, b_n, denoiser=model,
        isPred=False, original=im_original, verbose=False, device=device)
    xnp = torch.squeeze(torch.abs(x)).cpu().detach().numpy()
    res['PnP-ISTA'] = {'x': xnp, 'psnr': psnr, 'time': t, 'fixed': fixed,
                       'ssim': ssim_metric(np.abs(im_original), xnp, data_range=1.0)}

    # Pre1-PnP — fixed degree-1 Chebyshev preconditioner
    w1 = lambda xx: 2*xx - ATx_fn(Ax_fn(xx))
    x, psnr, t, fixed = opt.ISTA_PnP(
        n_iters, Ax_fn, ATx_fn, b_n, denoiser=model,
        isPred=True, w_pred=w1, original=im_original, verbose=False, device=device)
    xnp = torch.squeeze(torch.abs(x)).cpu().detach().numpy()
    res['Pre1-PnP'] = {'x': xnp, 'psnr': psnr, 'time': t, 'fixed': fixed,
                       'ssim': ssim_metric(np.abs(im_original), xnp, data_range=1.0)}

    # Dynamic PnP — quasi-Newton preconditioner (updated each iteration)
    x, psnr, t, fixed, _ = opt.DynamicPnP(
        n_iters, Ax_fn, ATx_fn, b_n, denoiser=model,
        original=im_original, verbose=False, device=device)
    xnp = torch.squeeze(torch.abs(x)).cpu().detach().numpy()
    res['Dynamic-PnP'] = {'x': xnp, 'psnr': psnr, 'time': t, 'fixed': fixed,
                          'ssim': ssim_metric(np.abs(im_original), xnp, data_range=1.0)}

    return res
```

---

## 3. Spiral Experiment

**Setup:** 32-arm spiral trajectory (`trj_full`). Acceleration varied by keeping every `step`-th arm → R ≈ 32 / n_arms.

```python
spiral_steps   = [2, 4, 6, 8, 16]
MaxIter_exp    = 100
spiral_results = {}

for step in spiral_steps:
    Ax_s, ATx_s, n_arms = build_spiral_ops(step)
    b_s = simulate_measurements(Ax_s, noise_level)
    spiral_results[step] = run_all_methods(Ax_s, ATx_s, b_s, MaxIter_exp)
    spiral_results[step]['R'] = 32 / n_arms
```

### Results

| Step | Arms | R | Method | PSNR (dB) | SSIM |
|------|------|------|--------------|-----------|------|
| 2 | 16 | 2.0× | PnP-ISTA | 39.30 | 0.8198 |
| 2 | 16 | 2.0× | Pre1-PnP | 42.17 | 0.8308 |
| 2 | 16 | 2.0× | Dynamic-PnP | **43.46** | **0.8368** |
| 4 | 8 | 4.0× | PnP-ISTA | 34.29 | 0.7687 |
| 4 | 8 | 4.0× | Pre1-PnP | 36.75 | 0.7818 |
| 4 | 8 | 4.0× | Dynamic-PnP | **39.59** | **0.7879** |
| 6 | 6 | 5.3× | PnP-ISTA | 32.28 | 0.7308 |
| 6 | 6 | 5.3× | Pre1-PnP | 34.46 | 0.7397 |
| 6 | 6 | 5.3× | Dynamic-PnP | **37.30** | **0.7416** |
| 8 | 4 | 8.0× | PnP-ISTA | 29.97 | 0.6871 |
| 8 | 4 | 8.0× | Pre1-PnP | 31.34 | **0.6878** |
| 8 | 4 | 8.0× | Dynamic-PnP | **32.56** | 0.6804 |
| 16 | 2 | 16.0× | PnP-ISTA | 26.14 | **0.5901** |
| 16 | 2 | 16.0× | Pre1-PnP | 26.75 | 0.5712 |
| 16 | 2 | 16.0× | Dynamic-PnP | **26.93** | 0.4701 |

**Observation:** Dynamic-PnP consistently achieves the highest PSNR across all spiral acceleration levels. At very high acceleration (R=16×), the SSIM gap between methods narrows and Dynamic-PnP's SSIM drops, suggesting potential instability at extreme undersampling.

---

## 4. Radial Experiment

**Setup:** Golden-angle radial trajectory generated in code. Nyquist ≈ 402 spokes for 256×256 → R ≈ 402 / nspokes.

```python
radial_spokes  = [32, 21, 16, 8, 4]
radial_results = {}

for nspokes in radial_spokes:
    Ax_r, ATx_r = build_radial_ops(nspokes)
    b_r = simulate_measurements(Ax_r, noise_level)
    radial_results[nspokes] = run_all_methods(Ax_r, ATx_r, b_r, MaxIter_exp)
    radial_results[nspokes]['R'] = int(np.pi / 2 * 256) / nspokes
```

### Results

| Spokes | R | Method | PSNR (dB) | SSIM |
|--------|------|--------------|-----------|------|
| 32 | 12.6× | PnP-ISTA | 31.76 | **0.7396** |
| 32 | 12.6× | Pre1-PnP | **32.66** | 0.6838 |
| 32 | 12.6× | Dynamic-PnP | 32.14 | 0.5659 |
| 21 | 19.1× | PnP-ISTA | 30.05 | **0.7325** |
| 21 | 19.1× | Pre1-PnP | **30.82** | 0.6926 |
| 21 | 19.1× | Dynamic-PnP | 29.44 | 0.5390 |
| 16 | 25.1× | PnP-ISTA | 28.81 | **0.7290** |
| 16 | 25.1× | Pre1-PnP | 29.41 | 0.6975 |
| 16 | 25.1× | Dynamic-PnP | **29.76** | 0.5930 |
| 8 | 50.2× | PnP-ISTA | 26.73 | **0.7107** |
| 8 | 50.2× | Pre1-PnP | **27.11** | 0.6947 |
| 8 | 50.2× | Dynamic-PnP | 26.98 | 0.5959 |
| 4 | 100.5× | PnP-ISTA | 25.14 | **0.6763** |
| 4 | 100.5× | Pre1-PnP | **25.42** | 0.6703 |
| 4 | 100.5× | Dynamic-PnP | 24.77 | 0.4740 |

**Observation:** For radial, Pre1-PnP tends to match or beat Dynamic-PnP in PSNR, while PnP-ISTA consistently achieves the highest SSIM. Dynamic-PnP shows notably lower SSIM across all radial settings, suggesting the quasi-Newton preconditioner is less well-suited to radial than spiral trajectories.

---

## 5. Evaluation Plots

```python
methods = ['PnP-ISTA', 'Pre1-PnP', 'Dynamic-PnP']
colors  = {'PnP-ISTA': 'steelblue', 'Pre1-PnP': 'darkorange', 'Dynamic-PnP': 'forestgreen'}
ls_map  = ['-', '--', ':', '-.', (0,(3,1,1,1))]
```

### 5.1 PSNR vs Iteration

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (results, steps, label) in zip(
    axes,
    [(spiral_results, spiral_steps, 'Spiral'), (radial_results, radial_spokes, 'Radial')]
):
    for method in methods:
        for i, key in enumerate(steps):
            ax.plot(results[key][method]['psnr'],
                    color=colors[method], linestyle=ls_map[i],
                    label=f"{method} R≈{results[key]['R']:.1f}x")
    ax.set_title(f'{label}: PSNR vs Iteration')
    ax.set_xlabel('Iteration'); ax.set_ylabel('PSNR (dB)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True)
plt.tight_layout(); plt.show()
```

### 5.2 PSNR vs Runtime

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (results, steps, label) in zip(
    axes,
    [(spiral_results, spiral_steps, 'Spiral'), (radial_results, radial_spokes, 'Radial')]
):
    for method in methods:
        for i, key in enumerate(steps):
            ax.plot(results[key][method]['time'], results[key][method]['psnr'],
                    color=colors[method], linestyle=ls_map[i],
                    label=f"{method} R≈{results[key]['R']:.1f}x")
    ax.set_title(f'{label}: PSNR vs Runtime')
    ax.set_xlabel('Cumulative time (s)'); ax.set_ylabel('PSNR (dB)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True)
plt.tight_layout(); plt.show()
```

### 5.3 Final PSNR and SSIM vs Acceleration

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for col, (results, steps, label) in enumerate(
    [(spiral_results, spiral_steps, 'Spiral'), (radial_results, radial_spokes, 'Radial')]
):
    R_vals = [results[k]['R'] for k in steps]
    for method in methods:
        axes[0, col].plot(R_vals, [results[k][method]['psnr'][-1] for k in steps],
                          marker='o', color=colors[method], label=method)
        axes[1, col].plot(R_vals, [results[k][method]['ssim'] for k in steps],
                          marker='o', color=colors[method], label=method)
    axes[0, col].set_title(f'{label}: Final PSNR vs R')
    axes[0, col].set_xlabel('Acceleration R'); axes[0, col].set_ylabel('PSNR (dB)')
    axes[0, col].legend(); axes[0, col].grid(True)
    axes[1, col].set_title(f'{label}: SSIM vs R')
    axes[1, col].set_xlabel('Acceleration R'); axes[1, col].set_ylabel('SSIM')
    axes[1, col].legend(); axes[1, col].grid(True)
plt.tight_layout(); plt.show()
```

### 5.4 Fixed-Point Residual (Stability)

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, (results, steps, label) in zip(
    axes,
    [(spiral_results, spiral_steps, 'Spiral'), (radial_results, radial_spokes, 'Radial')]
):
    for method in methods:
        for i, key in enumerate(steps):
            ax.semilogy(results[key][method]['fixed'],
                        color=colors[method], linestyle=ls_map[i],
                        label=f"{method} R≈{results[key]['R']:.1f}x")
    ax.set_title(f'{label}: Fixed-Point Residual\n||x - D(x - ∇f(x))||')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Residual (log scale)')
    ax.legend(fontsize=7, ncol=2); ax.grid(True)
plt.tight_layout(); plt.show()
```

> **Interpreting stability:** A residual that steadily decreases to near zero indicates convergence (stable). A residual that oscillates or grows at high R indicates instability.

---

## 6. Summary

| Trajectory | R range | Best PSNR method | Best SSIM method |
|------------|---------|-----------------|-----------------|
| Spiral | 2× – 16× | Dynamic-PnP (low-mid R) | Dynamic-PnP (low-mid R) |
| Radial | 12.6× – 100.5× | Pre1-PnP | PnP-ISTA |

- **Dynamic-PnP** leads on spiral at moderate acceleration but loses its SSIM advantage at R=16×
- **Pre1-PnP** offers consistent gains over standard PnP-ISTA with low overhead (one extra `ATx∘Ax` per iteration)
- **PnP-ISTA** is the most structurally stable (best SSIM on radial), though lowest PSNR
- At extreme acceleration (R > 50×), all methods converge to similar PSNR, with differences mainly in SSIM and fixed-point residual behavior
