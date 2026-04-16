"""
dataset.py
----------
Loaders for the MRI datasets used in the P²nP paper.

Classes
-------
BrainMRIDataset
    Reads the MoDL brain MRI data from ``data/dataset.hdf5``.
    Note: despite the 256×232 aspect ratio, visual inspection confirms
    these are sagittal brain images (not knee).  Good test indices are
    in the range 40–120 (edge slices near 0 and 163 are mostly empty).

FastMRIBrainDataset
    Reads fastMRI brain multi-coil k-space files (``file_brain_*.h5``).
    Each file contains multiple slices; samples are indexed flat across
    all files.

Brain HDF5 structure (MoDL)
----------------------------
  trnOrg  : (360, 256, 232)     complex64  – training ground-truth images
  trnCsm  : (360, 12, 256, 232) complex64  – training coil sensitivity maps
  trnMask : (360, 256, 232)     int8        – training sampling masks
  tstOrg  : (164, 256, 232)     complex64  – test ground-truth images
  tstCsm  : (164, 12, 256, 232) complex64  – test coil sensitivity maps
  tstMask : (164, 256, 232)     int8        – test sampling masks

Brain H5 structure (per file, fastMRI format)
---------------------------------------------
  kspace : (n_slices, n_coils, H, W) complex64  – fully-sampled k-space
  mask   : (W,)                      float32    – 1D Cartesian undersampling mask
  attrs  : acceleration, acquisition, num_low_frequency, patient_id

Data sources
------------
  Brain (MoDL):
    Aggarwal, H. K. (2019). MoDL: Model-Based Deep Learning Dataset [Data set].
    In IEEE Transactions on Medical Imaging (Vol. 38, Number 2, pp. 394–405).
    Zenodo. https://doi.org/10.5281/zenodo.1199582

  Brain (fastMRI):
    Zbontar, J., et al. (2018). fastMRI: An Open Dataset and Benchmarks for
    Accelerated MRI. arXiv:1811.08839.
    https://fastmri.org
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import scipy.io
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Public data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MRISample:
    """A single knee MRI sample with its associated metadata."""

    index: int
    split: Literal["train", "test"]
    image: np.ndarray          # complex64, shape (H, W)
    csm: np.ndarray            # complex64, shape (n_coils, H, W)
    mask: np.ndarray           # int8,      shape (H, W)

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.image)

    @property
    def n_coils(self) -> int:
        return self.csm.shape[0]

    def to_mat_dict(self) -> dict:
        """Return a dict compatible with scipy.io.savemat / loadmat."""
        return {
            "im_real": self.image.real,
            "im_imag": self.image.imag,
        }


@dataclass(frozen=True)
class BrainMRISample:
    """A single brain MRI sample reconstructed from fastMRI k-space data."""

    flat_index: int            # index into the dataset's flat slice list
    file_path: Path            # source .h5 file
    slice_index: int           # slice within the file
    image: np.ndarray          # complex64, shape (H, W) – coil-combined GT
    csm: np.ndarray            # complex64, shape (n_coils, H, W) – estimated CSM
    mask_1d: np.ndarray        # float32,   shape (W_orig,) – fastMRI column mask

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.image)

    @property
    def n_coils(self) -> int:
        return self.csm.shape[0]

    def to_mat_dict(self) -> dict:
        """Return a dict compatible with scipy.io.savemat / loadmat."""
        return {
            "im_real": self.image.real,
            "im_imag": self.image.imag,
        }


# ---------------------------------------------------------------------------
# Brain dataset (MoDL HDF5)
# ---------------------------------------------------------------------------

class BrainMRIDataset:
    """
    Read-only interface to the P²nP brain MRI HDF5 dataset (MoDL).

    The dataset contains sagittal brain MRI slices (256×232 px, 12 coils).
    Edge slices (indices near 0 and 163) are mostly empty — use indices
    in the range 40–120 for meaningful reconstructions.

    Parameters
    ----------
    hdf5_path : str | Path
        Path to ``dataset.hdf5``.
    target_size : tuple[int, int]
        Spatial size (H, W) to crop/resize images to.  Defaults to (256, 256)
        to match the paper's pre-processing.
    normalize : bool
        If True (default), scale each image so that max |image| == 1.

    Examples
    --------
    >>> ds = BrainMRIDataset("data/dataset.hdf5")
    >>> sample = ds.get(40, split="test")
    >>> print(sample.image.shape)   # (256, 256)
    >>> ds.save_mat(40, split="test", out_dir="image")
    """

    _SPLIT_KEYS = {
        "train": ("trnOrg", "trnCsm", "trnMask"),
        "test":  ("tstOrg", "tstCsm", "tstMask"),
    }

    def __init__(
        self,
        hdf5_path: str | Path,
        target_size: tuple[int, int] = (256, 256),
        normalize: bool = True,
    ) -> None:
        self.hdf5_path = Path(hdf5_path)
        self.target_size = target_size
        self.normalize = normalize

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        with h5py.File(self.hdf5_path, "r") as f:
            self._n_train: int = f["trnOrg"].shape[0]
            self._n_test: int  = f["tstOrg"].shape[0]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_train(self) -> int:
        return self._n_train

    @property
    def n_test(self) -> int:
        return self._n_test

    def __len__(self) -> int:
        return self._n_train + self._n_test

    def __repr__(self) -> str:
        return (
            f"BrainMRIDataset("
            f"path='{self.hdf5_path}', "
            f"train={self.n_train}, "
            f"test={self.n_test}, "
            f"target_size={self.target_size})"
        )

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def get(self, index: int, split: Literal["train", "test"] = "test") -> MRISample:
        """
        Load a single sample by index.

        Parameters
        ----------
        index : int
            Zero-based index within the chosen split.
        split : "train" | "test"
            Which split to load from.

        Returns
        -------
        MRISample
        """
        n = self.n_train if split == "train" else self.n_test
        if not (0 <= index < n):
            raise IndexError(f"Index {index} out of range for {split} split (size {n}).")

        org_key, csm_key, mask_key = self._SPLIT_KEYS[split]

        with h5py.File(self.hdf5_path, "r") as f:
            image = f[org_key][index]   # (H, W) complex64
            csm   = f[csm_key][index]   # (n_coils, H, W) complex64
            mask  = f[mask_key][index]  # (H, W) int8

        image = self._preprocess_image(image)
        csm   = self._preprocess_csm(csm)
        mask  = self._preprocess_mask(mask)

        return MRISample(index=index, split=split, image=image, csm=csm, mask=mask)

    def get_batch(
        self,
        indices: list[int],
        split: Literal["train", "test"] = "test",
    ) -> list[MRISample]:
        """Load multiple samples at once."""
        return [self.get(i, split=split) for i in indices]

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def save_mat(
        self,
        index: int,
        split: Literal["train", "test"] = "test",
        out_dir: str | Path = "image",
        filename: str | None = None,
    ) -> Path:
        """
        Save a sample as a .mat file compatible with the notebook's loadmat call.

        The file contains ``im_real`` and ``im_imag`` arrays.

        Parameters
        ----------
        index : int
            Zero-based index within the split.
        split : "train" | "test"
        out_dir : str | Path
            Directory to write into (created if missing).
        filename : str | None
            Override the auto-generated name ``KneeGT_{index+1}.mat``.

        Returns
        -------
        Path
            Path of the written file.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sample = self.get(index, split=split)

        if filename is None:
            filename = f"BrainDeepGT_{index + 1}.mat"

        out_path = out_dir / filename
        scipy.io.savemat(str(out_path), sample.to_mat_dict())
        return out_path

    def export_test_set(
        self,
        indices: list[int],
        out_dir: str | Path = "image",
    ) -> list[Path]:
        """
        Export a list of test images as .mat files.

        Parameters
        ----------
        indices : list[int]
            Zero-based indices to export (e.g. [0,1,2,3,4,5] for the first 6).
        out_dir : str | Path
            Output directory.

        Returns
        -------
        list[Path]
            Paths of the written files.
        """
        return [self.save_mat(i, split="test", out_dir=out_dir) for i in indices]

    # ------------------------------------------------------------------
    # Pre-processing (mirrors the paper's pipeline)
    # ------------------------------------------------------------------

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Crop/resize to target_size and optionally normalize."""
        image = self._resize_complex(image, self.target_size)
        if self.normalize:
            max_val = np.max(np.abs(image))
            if max_val > 0:
                image = image / max_val
        return image.astype(np.complex64)

    def _preprocess_csm(self, csm: np.ndarray) -> np.ndarray:
        """Resize each coil sensitivity map to target_size."""
        resized = np.stack(
            [self._resize_complex(csm[c], self.target_size) for c in range(csm.shape[0])],
            axis=0,
        )
        return resized.astype(np.complex64)

    def _preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        h, w = self.target_size
        return mask[:h, :w].astype(np.int8)

    @staticmethod
    def _resize_complex(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Resize a complex array by resizing real and imaginary parts separately."""
        h, w = size
        if arr.shape == (h, w):
            return arr
        real = resize(arr.real, (h, w), anti_aliasing=True, preserve_range=True)
        imag = resize(arr.imag, (h, w), anti_aliasing=True, preserve_range=True)
        return (real + 1j * imag).astype(arr.dtype)


# ---------------------------------------------------------------------------
# Knee dataset (MoDL HDF5)
# ---------------------------------------------------------------------------

class KneeMRIDataset(BrainMRIDataset):
    """
    Read-only interface to a MoDL-format knee MRI HDF5 dataset.

    Identical structure to BrainMRIDataset — just point it at a knee HDF5
    file (same trnOrg / tstOrg / trnCsm / tstCsm / trnMask / tstMask keys).

    All slices tend to contain anatomy (no empty edge slices like the brain
    dataset), so any index in [0, n_test) is usable.

    Parameters
    ----------
    hdf5_path : str | Path
        Path to the knee ``dataset.hdf5``.
    target_size : tuple[int, int]
        Defaults to (256, 256).
    normalize : bool
        If True (default), scale each image so that max |image| == 1.

    Examples
    --------
    >>> ds = KneeMRIDataset("data/knee_dataset.hdf5")
    >>> sample = ds.get(0, split="test")
    >>> print(sample.image.shape)   # (256, 256)
    >>> ds.save_mat(0, split="test", out_dir="image")
    """

    def __repr__(self) -> str:
        return (
            f"KneeMRIDataset("
            f"path='{self.hdf5_path}', "
            f"train={self.n_train}, "
            f"test={self.n_test}, "
            f"target_size={self.target_size})"
        )

    def save_mat(
        self,
        index: int,
        split: Literal["train", "test"] = "test",
        out_dir: str | Path = "image",
        filename: str | None = None,
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sample = self.get(index, split=split)
        if filename is None:
            filename = f"KneeGT_{index + 1}.mat"
        out_path = out_dir / filename
        scipy.io.savemat(str(out_path), sample.to_mat_dict())
        return out_path


# ---------------------------------------------------------------------------
# Knee dataset (fastMRI single-coil H5)
# ---------------------------------------------------------------------------

class FastMRIKneeDataset:
    """
    Read-only interface to a fastMRI knee .h5 file.

    Handles both single-coil (3-D kspace: slices × H × W) and multi-coil
    (4-D kspace: slices × coils × H × W) knee files automatically.

    - Single-coil: GT image obtained directly via centered 2D IFFT.
    - Multi-coil:  GT obtained via per-coil IFFT, RSS-based CSM estimation,
      and coil combination (same pipeline as FastMRIBrainDataset).

    Parameters
    ----------
    h5_path : str | Path
        Path to the knee ``.h5`` file (e.g. ``data/file1001423.h5``).
    target_size : tuple[int, int]
        Spatial size (H, W) of the output image.  Defaults to (256, 256).
    center_crop : bool
        If True (default), center-crop to ``target_size`` after IFFT, preserving
        native resolution.  If False, resize (bilinear) — causes blurring when
        the native size (e.g. 640×368) is much larger than ``target_size``.
    normalize : bool
        If True (default), scale each image so that max |image| == 1.

    Examples
    --------
    >>> ds = FastMRIKneeDataset("data/file1001423.h5")   # single-coil, crop
    >>> ds = FastMRIKneeDataset("data/file1000986.h5")   # multi-coil (15 coils), crop
    >>> print(len(ds))              # number of slices
    >>> sample = ds.get(10)
    >>> print(sample.image.shape)   # (256, 256)
    >>> ds.save_mat(10, out_dir="image")   # → image/KneeGT_11.mat
    """

    def __init__(
        self,
        h5_path: str | Path,
        target_size: tuple[int, int] = (256, 256),
        center_crop: bool = False,
        espirit: bool = False,
        normalize: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.target_size = target_size
        self.center_crop = center_crop
        self.espirit = espirit
        self.normalize = normalize

        if not self.h5_path.exists():
            raise FileNotFoundError(f"File not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as f:
            kspace_shape = f["kspace"].shape
            self._n_slices: int = kspace_shape[0]
            self._mask_1d: np.ndarray = f["mask"][:] if "mask" in f else np.ones(kspace_shape[-1], dtype=bool)
            self._attrs: dict = dict(f.attrs)

        # Detect single-coil (3D) vs multi-coil (4D)
        self._multi_coil: bool = len(kspace_shape) == 4
        self._n_coils: int = kspace_shape[1] if self._multi_coil else 1

    def __len__(self) -> int:
        return self._n_slices

    def __repr__(self) -> str:
        coil_str = f"{self._n_coils} coils" if self._multi_coil else "single-coil"
        pp_str = "crop" if self.center_crop else "resize"
        return (
            f"FastMRIKneeDataset("
            f"path='{self.h5_path}', "
            f"slices={self._n_slices}, "
            f"{coil_str}, "
            f"preprocess='{pp_str}', "
            f"target_size={self.target_size}, "
            f"acquisition='{self._attrs.get('acquisition', 'unknown')}')"
        )

    def get(self, index: int) -> BrainMRISample:
        """
        Load and reconstruct a single slice by index.

        For single-coil files returns a dummy unit CSM (1 coil).
        For multi-coil files returns the estimated RSS-based CSM.
        """
        if not (0 <= index < self._n_slices):
            raise IndexError(
                f"Index {index} out of range (file has {self._n_slices} slices)."
            )

        with h5py.File(self.h5_path, "r") as f:
            kspace_slice = f["kspace"][index]   # (H, W) or (n_coils, H, W)

        _process = self._center_crop if self.center_crop else self._resize_complex

        if self._multi_coil:
            # kspace_slice: (n_coils, H, W)
            img_coils = self._ifft2c(kspace_slice)          # (n_coils, H, W)
            if self.espirit:
                csm = self._estimate_csm_espirit(kspace_slice)  # (n_coils, H, W)
            else:
                csm = self._estimate_csm(img_coils)             # (n_coils, H, W)
            image = np.sum(np.conj(csm) * img_coils, axis=0)  # (H, W)
            image = _process(image, self.target_size)
            csm = np.stack(
                [_process(csm[c], self.target_size) for c in range(csm.shape[0])],
                axis=0,
            ).astype(np.complex64)
        else:
            # kspace_slice: (H, W)
            image = self._ifft2c(kspace_slice)              # (H, W)
            image = _process(image, self.target_size)
            csm = np.ones((1, *self.target_size), dtype=np.complex64)

        if self.normalize:
            max_val = np.max(np.abs(image))
            if max_val > 0:
                image = image / max_val

        return BrainMRISample(
            flat_index=index,
            file_path=self.h5_path,
            slice_index=index,
            image=image.astype(np.complex64),
            csm=csm,
            mask_1d=self._mask_1d,
        )

    def get_batch(self, indices: list[int]) -> list[BrainMRISample]:
        return [self.get(i) for i in indices]

    def save_mat(
        self,
        index: int,
        out_dir: str | Path = "image",
        filename: str | None = None,
    ) -> Path:
        """Save slice as ``KneeGT_{index+1}.mat`` with im_real / im_imag."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sample = self.get(index)
        if filename is None:
            filename = f"KneeGT_{index + 1}.mat"
        out_path = out_dir / filename
        scipy.io.savemat(str(out_path), sample.to_mat_dict())
        return out_path

    def export_set(
        self,
        indices: list[int],
        out_dir: str | Path = "image",
    ) -> list[Path]:
        return [self.save_mat(i, out_dir=out_dir) for i in indices]

    @staticmethod
    def _ifft2c(kspace: np.ndarray) -> np.ndarray:
        """Centered 2D IFFT applied to the last two axes (ortho normalization)."""
        return np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(kspace, axes=(-2, -1)),
                axes=(-2, -1),
                norm="ortho",
            ),
            axes=(-2, -1),
        ).astype(np.complex64)

    @staticmethod
    def _estimate_csm(img_coils: np.ndarray) -> np.ndarray:
        """RSS-based coil sensitivity estimation (same as FastMRIBrainDataset)."""
        rss = np.sqrt(np.sum(np.abs(img_coils) ** 2, axis=0, keepdims=True)) + 1e-8
        return (img_coils / rss).astype(np.complex64)

    @staticmethod
    def _estimate_csm_espirit(kspace: np.ndarray, calib_width: int = 24) -> np.ndarray:
        """
        ESPIRiT coil sensitivity estimation via sigpy.

        Parameters
        ----------
        kspace : (n_coils, H, W) complex64
            Fully-sampled multi-coil k-space for one slice.
        calib_width : int
            Size of the auto-calibration signal (ACS) region.  Default 24.

        Returns
        -------
        csm : (n_coils, H, W) complex64
        """
        import sigpy.mri as mr
        # sigpy ESPIRiT expects (n_coils, H, W), returns (n_maps, n_coils, H, W)
        # We take the first map (dominant eigenmap)
        csm = mr.app.EspiritCalib(
            kspace.astype(np.complex64),
            calib_width=calib_width,
            show_pbar=False,
        ).run()
        if csm.ndim == 4:
            csm = csm[0]   # (n_coils, H, W)
        return csm.astype(np.complex64)

    @staticmethod
    def _center_crop(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Center-crop a 2D complex array to (H, W) without any interpolation."""
        h, w = size
        ah, aw = arr.shape[-2], arr.shape[-1]
        r0 = max((ah - h) // 2, 0)
        c0 = max((aw - w) // 2, 0)
        cropped = arr[..., r0:r0 + h, c0:c0 + w]
        # If the image is smaller than target, pad with zeros
        if cropped.shape[-2] < h or cropped.shape[-1] < w:
            pad_h = h - cropped.shape[-2]
            pad_w = w - cropped.shape[-1]
            cropped = np.pad(
                cropped,
                ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2)),
                mode='constant',
            )
        return cropped.astype(arr.dtype)

    @staticmethod
    def _resize_complex(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        h, w = size
        if arr.shape == (h, w):
            return arr
        real = resize(arr.real, (h, w), anti_aliasing=True, preserve_range=True)
        imag = resize(arr.imag, (h, w), anti_aliasing=True, preserve_range=True)
        return (real + 1j * imag).astype(arr.dtype)


# ---------------------------------------------------------------------------
# Brain dataset (fastMRI H5 files)
# ---------------------------------------------------------------------------

class FastMRIBrainDataset:
    """
    Read-only interface to a directory of fastMRI brain .h5 files.

    Each .h5 file stores the fully-sampled multi-coil k-space for multiple
    axial slices.  This class builds a flat index over all (file, slice) pairs
    so that each sample can be accessed with a single integer index.

    The ground-truth complex image is reconstructed from the full k-space via:
      1. Centered 2D IFFT per coil.
      2. RSS-based coil sensitivity estimation.
      3. Coil combination:  image = Σ_c  conj(csm_c) · img_c

    Parameters
    ----------
    data_dir : str | Path
        Directory containing ``file_brain_*.h5`` files.
    target_size : tuple[int, int]
        Spatial size (H, W) to resize images/CSM to.  Defaults to (256, 256).
    normalize : bool
        If True (default), scale each image so that max |image| == 1.

    Examples
    --------
    >>> ds = FastMRIBrainDataset("data/")
    >>> print(len(ds))                        # total slices across all files
    >>> sample = ds.get(0)
    >>> print(sample.image.shape)             # (256, 256)
    >>> ds.save_mat(0, out_dir="image")       # → image/BrainGT_1.mat
    >>> ds.export_set([0,1,2,3,4,5], "image") # export first 6 slices
    """

    def __init__(
        self,
        data_dir: str | Path,
        target_size: tuple[int, int] = (256, 256),
        normalize: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.normalize = normalize

        self._files: list[Path] = sorted(self.data_dir.glob("file_brain_*.h5"))
        if not self._files:
            raise FileNotFoundError(
                f"No 'file_brain_*.h5' files found in {self.data_dir}"
            )

        # Build flat index: list of (file_idx, slice_idx) tuples
        self._index: list[tuple[int, int]] = []
        self._slices_per_file: list[int] = []
        for fi, fpath in enumerate(self._files):
            with h5py.File(fpath, "r") as f:
                n_slices = f["kspace"].shape[0]
            self._slices_per_file.append(n_slices)
            for si in range(n_slices):
                self._index.append((fi, si))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return (
            f"FastMRIBrainDataset("
            f"dir='{self.data_dir}', "
            f"files={len(self._files)}, "
            f"total_slices={len(self)}, "
            f"target_size={self.target_size})"
        )

    def file_info(self) -> list[dict]:
        """Return a summary list of {file, n_slices} for each H5 file."""
        return [
            {"file": self._files[i].name, "n_slices": n}
            for i, n in enumerate(self._slices_per_file)
        ]

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def get(self, index: int) -> BrainMRISample:
        """
        Load and reconstruct a single sample by flat index.

        Parameters
        ----------
        index : int
            Zero-based index across all files and slices.

        Returns
        -------
        BrainMRISample
        """
        if not (0 <= index < len(self)):
            raise IndexError(
                f"Index {index} out of range (dataset has {len(self)} slices)."
            )

        fi, si = self._index[index]
        fpath = self._files[fi]

        with h5py.File(fpath, "r") as f:
            kspace = f["kspace"][si]   # (n_coils, H, W) complex64
            mask_1d = f["mask"][:]     # (W,) float32

        # Reconstruct coil images via centered 2D IFFT
        img_coils = self._ifft2c(kspace)           # (n_coils, H, W)

        # Estimate coil sensitivity maps from RSS
        csm = self._estimate_csm(img_coils)        # (n_coils, H, W)

        # Coil combination: x = Σ conj(csm) * img
        image = np.sum(np.conj(csm) * img_coils, axis=0)  # (H, W)

        # Resize to target_size
        image = self._resize_complex(image, self.target_size)
        csm = np.stack(
            [self._resize_complex(csm[c], self.target_size) for c in range(csm.shape[0])],
            axis=0,
        ).astype(np.complex64)

        if self.normalize:
            max_val = np.max(np.abs(image))
            if max_val > 0:
                image = image / max_val

        return BrainMRISample(
            flat_index=index,
            file_path=fpath,
            slice_index=si,
            image=image.astype(np.complex64),
            csm=csm,
            mask_1d=mask_1d,
        )

    def get_batch(self, indices: list[int]) -> list[BrainMRISample]:
        """Load multiple samples at once."""
        return [self.get(i) for i in indices]

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def save_mat(
        self,
        index: int,
        out_dir: str | Path = "image",
        filename: str | None = None,
    ) -> Path:
        """
        Save a sample as a .mat file (``im_real`` + ``im_imag``).

        Parameters
        ----------
        index : int
            Flat index into the dataset.
        out_dir : str | Path
            Output directory (created if missing).
        filename : str | None
            Override auto-generated name ``BrainGT_{index+1}.mat``.

        Returns
        -------
        Path
            Path of the written file.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        sample = self.get(index)

        if filename is None:
            filename = f"BrainGT_{index + 1}.mat"

        out_path = out_dir / filename
        scipy.io.savemat(str(out_path), sample.to_mat_dict())
        return out_path

    def export_set(
        self,
        indices: list[int],
        out_dir: str | Path = "image",
    ) -> list[Path]:
        """
        Export a list of slices as .mat files.

        Parameters
        ----------
        indices : list[int]
            Flat indices to export (e.g. [0,1,2,3,4,5] for first 6 slices).
        out_dir : str | Path
            Output directory.

        Returns
        -------
        list[Path]
            Paths of the written files.
        """
        return [self.save_mat(i, out_dir=out_dir) for i in indices]

    # ------------------------------------------------------------------
    # Signal processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ifft2c(kspace: np.ndarray) -> np.ndarray:
        """Centered 2D IFFT applied to the last two axes (ortho normalization)."""
        return np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(kspace, axes=(-2, -1)),
                axes=(-2, -1),
                norm="ortho",
            ),
            axes=(-2, -1),
        ).astype(np.complex64)

    @staticmethod
    def _estimate_csm(img_coils: np.ndarray) -> np.ndarray:
        """
        Estimate coil sensitivity maps via RSS normalisation.

        Each coil image is divided by the root-sum-of-squares across coils,
        giving a unit-norm sensitivity estimate.
        """
        rss = np.sqrt(np.sum(np.abs(img_coils) ** 2, axis=0, keepdims=True)) + 1e-8
        return (img_coils / rss).astype(np.complex64)

    @staticmethod
    def _resize_complex(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Resize a complex 2-D array (real and imaginary parts independently)."""
        h, w = size
        if arr.shape == (h, w):
            return arr
        real = resize(arr.real, (h, w), anti_aliasing=True, preserve_range=True)
        imag = resize(arr.imag, (h, w), anti_aliasing=True, preserve_range=True)
        return (real + 1j * imag).astype(arr.dtype)
