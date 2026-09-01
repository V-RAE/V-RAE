from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from scipy import linalg

CovarianceConvention = Literal["sample", "population"]
FrechetImplementation = Literal["scipy_eigh", "scipy_sqrtm", "torch_svd"]


def feature_statistics(
    features: torch.Tensor | np.ndarray,
    *,
    covariance: CovarianceConvention = "sample",
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        features.detach().cpu() if torch.is_tensor(features) else features, dtype=np.float64
    )
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Feature matrix must be [N,D] with N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("Features contain non-finite values")
    if covariance not in {"sample", "population"}:
        raise ValueError("covariance must be sample or population")
    mean = values.mean(axis=0)
    centered = values - mean
    denominator = values.shape[0] - 1 if covariance == "sample" else values.shape[0]
    matrix = centered.T @ centered / denominator
    return mean, matrix


def frechet_distance(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> float:
    mean_a = np.asarray(mean_a, dtype=np.float64)
    mean_b = np.asarray(mean_b, dtype=np.float64)
    covariance_a = np.asarray(covariance_a, dtype=np.float64)
    covariance_b = np.asarray(covariance_b, dtype=np.float64)
    if mean_a.shape != mean_b.shape or covariance_a.shape != covariance_b.shape:
        raise ValueError("Frechet statistics have incompatible shapes")
    if covariance_a.shape != (mean_a.size, mean_a.size):
        raise ValueError("Covariance shape does not match the feature dimension")
    covariance_a = (covariance_a + covariance_a.T) * 0.5
    covariance_b = (covariance_b + covariance_b.T) * 0.5

    def psd_eigenvalues(matrix: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigenvectors = linalg.eigh(matrix, check_finite=True)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        if float(eigenvalues.min(initial=0.0)) < -eps * scale:
            raise ValueError(f"{name} is not positive semidefinite")
        return np.clip(eigenvalues, 0.0, None), eigenvectors

    eigenvalues_a, eigenvectors_a = psd_eigenvalues(covariance_a, "covariance_a")
    square_root_a = (eigenvectors_a * np.sqrt(eigenvalues_a)) @ eigenvectors_a.T
    middle = square_root_a @ covariance_b @ square_root_a
    middle = (middle + middle.T) * 0.5
    middle_eigenvalues, _ = psd_eigenvalues(middle, "covariance product")
    trace_product_root = float(np.sqrt(middle_eigenvalues).sum())
    difference = mean_a - mean_b
    value = (
        difference @ difference
        + np.trace(covariance_a)
        + np.trace(covariance_b)
        - 2.0 * trace_product_root
    )
    return float(max(value, 0.0))


def frechet_from_features(
    real: torch.Tensor | np.ndarray,
    fake: torch.Tensor | np.ndarray,
    *,
    covariance: CovarianceConvention,
    implementation: FrechetImplementation = "scipy_eigh",
) -> float:
    if implementation == "torch_svd":
        return _torch_svd_frechet_from_features(real, fake, covariance=covariance)
    if implementation == "scipy_sqrtm":
        return _scipy_sqrtm_frechet_from_features(real, fake, covariance=covariance)
    if implementation != "scipy_eigh":
        raise ValueError("implementation must be scipy_eigh, scipy_sqrtm, or torch_svd")
    real_mean, real_covariance = feature_statistics(real, covariance=covariance)
    fake_mean, fake_covariance = feature_statistics(fake, covariance=covariance)
    return frechet_distance(real_mean, real_covariance, fake_mean, fake_covariance)


def _scipy_sqrtm_frechet_from_features(
    real: torch.Tensor | np.ndarray,
    fake: torch.Tensor | np.ndarray,
    *,
    covariance: CovarianceConvention,
) -> float:
    """Match the uni-vug/StyleGAN-V UCF101 FVD SciPy solver."""

    real_mean, real_covariance = feature_statistics(real, covariance=covariance)
    fake_mean, fake_covariance = feature_statistics(fake, covariance=covariance)
    mean_term = np.square(fake_mean - real_mean).sum()
    product_root, _ = linalg.sqrtm(fake_covariance @ real_covariance, disp=False)
    value = np.real(mean_term + np.trace(fake_covariance + real_covariance - 2.0 * product_root))
    return float(value)


def _torch_svd_square_root(matrix: torch.Tensor, *, eps: float = 1.0e-10) -> torch.Tensor:
    """Compute the matrix square root with the locked Torch-SVD procedure."""

    left, singular_values, right = torch.svd(matrix)
    roots = torch.where(singular_values < eps, singular_values, torch.sqrt(singular_values))
    return left @ torch.diag(roots) @ right.t()


def _torch_svd_frechet_from_features(
    fake: torch.Tensor | np.ndarray,
    real: torch.Tensor | np.ndarray,
    *,
    covariance: CovarianceConvention,
) -> float:
    """Compute Fréchet distance with float64 statistics and Torch SVD."""

    fake_values = torch.as_tensor(fake, dtype=torch.float64, device="cpu").flatten(1)
    real_values = torch.as_tensor(real, dtype=torch.float64, device="cpu").flatten(1)
    if fake_values.shape != real_values.shape or fake_values.ndim != 2:
        raise ValueError("Real/fake feature shapes must be equal [N,D]")
    if fake_values.shape[0] < 2:
        raise ValueError("At least two feature vectors are required")
    if not torch.isfinite(fake_values).all() or not torch.isfinite(real_values).all():
        raise ValueError("Features contain non-finite values")
    if covariance not in {"sample", "population"}:
        raise ValueError("covariance must be sample or population")

    denominator = fake_values.shape[0] - 1 if covariance == "sample" else fake_values.shape[0]
    factor = 1.0 / float(denominator)

    def statistics(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = values.mean(dim=0)
        centered = values - mean
        return mean, factor * centered.t().matmul(centered)

    fake_mean, fake_covariance = statistics(fake_values)
    real_mean, real_covariance = statistics(real_values)
    fake_root = _torch_svd_square_root(fake_covariance)
    covariance_product = torch.matmul(
        fake_root,
        torch.matmul(real_covariance, fake_root),
    )
    sqrt_trace_component = torch.trace(_torch_svd_square_root(covariance_product))
    trace_term = torch.trace(fake_covariance + real_covariance) - 2.0 * sqrt_trace_component
    mean_term = torch.sum((fake_mean - real_mean) ** 2)
    value = trace_term + mean_term
    return float(value.item())
