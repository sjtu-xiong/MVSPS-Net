"""Multivariate singularity-power-spectrum guided dual-stream network."""

from __future__ import annotations

import math

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from torchvision.models import resnet18


def local_holder_exponents(
    image: np.ndarray,
    *,
    wavelet: str = "db3",
    levels: int | None = None,
    neighborhood: int = 3,
) -> np.ndarray:
    """Estimate a local Holder-exponent map with 2-D wavelet leaders."""

    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2-D image, got shape {image.shape}")
    height, width = image.shape
    if min(height, width) < 2:
        raise ValueError("Both spatial dimensions must be at least 2")

    wavelet_obj = pywt.Wavelet(wavelet)
    max_levels = pywt.dwt_max_level(min(height, width), wavelet_obj.dec_len)
    if max_levels < 1:
        raise ValueError(
            f"Input {image.shape} is too small for wavelet '{wavelet}'. "
            "Use larger images or a shorter wavelet."
        )

    automatic_levels = max(1, int(np.floor(np.log2(min(height, width)))) - 2)
    selected_levels = min(automatic_levels, max_levels) if levels is None else levels
    if not 1 <= selected_levels <= max_levels:
        raise ValueError(
            f"levels must be in [1, {max_levels}] for input {image.shape}; "
            f"got {selected_levels}"
        )

    coefficients = pywt.wavedec2(
        image,
        wavelet=wavelet_obj,
        level=selected_levels,
        mode="periodization",
    )
    leaders = []
    for horizontal, vertical, diagonal in coefficients[1:]:
        magnitude = np.maximum.reduce(
            [np.abs(horizontal), np.abs(vertical), np.abs(diagonal)]
        )
        if neighborhood > 1:
            magnitude = maximum_filter(
                magnitude,
                size=neighborhood,
                mode="nearest",
            )
        repeat_y = math.ceil(height / magnitude.shape[0])
        repeat_x = math.ceil(width / magnitude.shape[1])
        upsampled = np.repeat(np.repeat(magnitude, repeat_y, axis=0), repeat_x, axis=1)
        leaders.append(upsampled[:height, :width])

    leader_stack = np.stack(leaders, axis=-1)
    scale_slice = slice(1, selected_levels - 1) if selected_levels >= 4 else slice(None)
    selected_leaders = leader_stack[..., scale_slice]
    scales = np.arange(1, selected_levels + 1, dtype=np.float64)[scale_slice]

    tiny = np.finfo(np.float64).tiny
    log_leaders = np.log2(np.maximum(selected_leaders, tiny))
    design = -scales
    denominator = np.dot(design, design)
    holder = np.sum(design[None, None, :] * log_leaders, axis=-1) / denominator
    holder[~np.isfinite(holder)] = np.nan
    return holder


def multivariate_singularity_power_spectrum(
    image_hh: np.ndarray,
    image_vv: np.ndarray,
    *,
    num_bins: int = 10,
    normalize_by_count: bool = True,
    wavelet: str = "db3",
    levels: int | None = None,
    neighborhood: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the bivariate power spectrum and its pixel-to-bin mapping."""

    image_hh = np.asarray(image_hh, dtype=np.float64)
    image_vv = np.asarray(image_vv, dtype=np.float64)
    if image_hh.shape != image_vv.shape or image_hh.ndim != 2:
        raise ValueError(
            "The two inputs must be 2-D arrays with identical shapes; "
            f"got {image_hh.shape} and {image_vv.shape}"
        )
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")

    holder_hh = local_holder_exponents(
        image_hh,
        wavelet=wavelet,
        levels=levels,
        neighborhood=neighborhood,
    )
    holder_vv = local_holder_exponents(
        image_vv,
        wavelet=wavelet,
        levels=levels,
        neighborhood=neighborhood,
    )

    hh_flat = image_hh.ravel()
    vv_flat = image_vv.ravel()
    alpha_hh = holder_hh.ravel()
    alpha_vv = holder_vv.ravel()
    valid = (
        np.isfinite(hh_flat)
        & np.isfinite(vv_flat)
        & np.isfinite(alpha_hh)
        & np.isfinite(alpha_vv)
    )
    if not np.any(valid):
        raise ValueError("The input pair contains no valid pixels")

    hh_valid = hh_flat[valid]
    vv_valid = vv_flat[valid]
    alpha_hh_valid = alpha_hh[valid]
    alpha_vv_valid = alpha_vv[valid]
    tiny = np.finfo(np.float64).tiny

    def bin_indices(values: np.ndarray) -> np.ndarray:
        minimum = values.min()
        width = (values.max() - minimum) / num_bins
        width = max(float(width), tiny)
        return np.clip(np.floor((values - minimum) / width).astype(int), 0, num_bins - 1)

    bins_hh = bin_indices(alpha_hh_valid)
    bins_vv = bin_indices(alpha_vv_valid)
    flat_bins = bins_hh * num_bins + bins_vv
    total_bins = num_bins * num_bins

    counts = np.bincount(flat_bins, minlength=total_bins).reshape(num_bins, num_bins)
    power_hh = np.bincount(
        flat_bins,
        weights=np.square(hh_valid),
        minlength=total_bins,
    ).reshape(num_bins, num_bins)
    power_vv = np.bincount(
        flat_bins,
        weights=np.square(vv_valid),
        minlength=total_bins,
    ).reshape(num_bins, num_bins)
    spectrum = np.sqrt(power_hh * power_vv)
    if normalize_by_count:
        spectrum = np.divide(
            spectrum,
            counts,
            out=np.zeros_like(spectrum),
            where=counts > 0,
        )

    index_map_flat = np.zeros(image_hh.size, dtype=np.int64)
    index_map_flat[valid] = flat_bins
    index_map = index_map_flat.reshape(image_hh.shape)
    return spectrum, index_map, holder_hh, holder_vv


class MVSPSBlock(nn.Module):
    """Project joint singularity-domain context back to two spatial gates."""

    def __init__(
        self,
        num_bins: int = 10,
        *,
        spectrum_channels: int = 32,
        spectrum_embedding_dim: int = 8,
        pixel_channels: int = 16,
        temperature: float = 0.7,
        gate_scale: float = 0.3,
        smooth_kernel_size: int = 3,
        coverage_prior_weight: float = 0.02,
    ) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= gate_scale <= 1:
            raise ValueError("gate_scale must be between 0 and 1")
        if smooth_kernel_size not in (0, 1) and smooth_kernel_size % 2 == 0:
            raise ValueError("smooth_kernel_size must be 0, 1, or an odd integer")

        self.num_bins = num_bins
        self.spectrum_embedding_dim = spectrum_embedding_dim
        self.temperature = temperature
        self.gate_scale = gate_scale
        self.smooth_kernel_size = smooth_kernel_size
        self.coverage_prior_weight = coverage_prior_weight

        self.spectrum_features = nn.Sequential(
            nn.Conv2d(3, spectrum_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(spectrum_channels, spectrum_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.spectrum_gate = nn.Conv2d(spectrum_channels, 1, kernel_size=1)
        self.spectrum_embedding = nn.Conv2d(
            spectrum_channels,
            spectrum_embedding_dim,
            kernel_size=1,
        )
        self.spectrum_weight = nn.Parameter(torch.tensor(0.0))
        self.mean_alpha_weight = nn.Parameter(torch.tensor(0.0))
        self.alpha_difference_weight = nn.Parameter(torch.tensor(0.0))

        pixel_input_channels = spectrum_embedding_dim + 4

        def pixel_gate() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(pixel_input_channels, pixel_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(pixel_channels, pixel_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(pixel_channels, 1, kernel_size=1),
            )

        self.hh_gate = pixel_gate()
        self.vv_gate = pixel_gate()

    @staticmethod
    def _normalize_pixels(values: torch.Tensor) -> torch.Tensor:
        minimum = values.amin(dim=(1, 2), keepdim=True)
        maximum = values.amax(dim=(1, 2), keepdim=True)
        return (values - minimum) / (maximum - minimum + 1e-8)

    def _center_gate(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=(2, 3), keepdim=True)
        gate = 1.0 + self.gate_scale * (values / (mean + 1e-6) - 1.0)
        return gate.clamp(1.0 - self.gate_scale, 1.0 + self.gate_scale)

    def _smooth(self, gate: torch.Tensor) -> torch.Tensor:
        if self.smooth_kernel_size > 1:
            kernel = self.smooth_kernel_size
            return F.avg_pool2d(gate, kernel_size=kernel, stride=1, padding=kernel // 2)
        return gate

    def forward(
        self,
        spectrum: torch.Tensor,
        index_map: torch.Tensor,
        holder_hh: torch.Tensor,
        holder_vv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_bins, second_num_bins = spectrum.shape
        if num_bins != self.num_bins or second_num_bins != self.num_bins:
            raise ValueError(
                f"Expected spectra shaped [B, {self.num_bins}, {self.num_bins}], "
                f"got {tuple(spectrum.shape)}"
            )
        height, width = index_map.shape[-2:]
        device = spectrum.device

        spectrum_normalized = spectrum / (
            spectrum.amax(dim=(1, 2), keepdim=True) + 1e-8
        )
        coordinates = torch.linspace(0, 1, num_bins, device=device, dtype=spectrum.dtype)
        alpha_hh_plane = coordinates.view(1, 1, num_bins, 1)
        alpha_vv_plane = coordinates.view(1, 1, 1, num_bins)
        alpha_mean = 0.5 * (alpha_hh_plane + alpha_vv_plane)
        alpha_difference = torch.abs(alpha_hh_plane - alpha_vv_plane)
        alpha_mean = alpha_mean.expand(batch_size, 1, num_bins, num_bins)
        alpha_difference = alpha_difference.expand(batch_size, 1, num_bins, num_bins)

        spectrum_input = torch.cat(
            [spectrum_normalized.unsqueeze(1), alpha_mean, alpha_difference],
            dim=1,
        )
        spectrum_features = self.spectrum_features(spectrum_input)

        flat_indices = index_map.reshape(batch_size, -1).long().clamp(
            0,
            num_bins * num_bins - 1,
        )
        coverage_maps = []
        for sample_indices in flat_indices:
            counts = torch.bincount(
                sample_indices,
                minlength=num_bins * num_bins,
            ).to(dtype=spectrum.dtype)
            coverage = counts.reshape(num_bins, num_bins) / float(height * width)
            inverse_coverage = 1.0 - coverage / (coverage.max() + 1e-8)
            coverage_maps.append(inverse_coverage - inverse_coverage.mean())
        coverage_prior = torch.stack(coverage_maps)

        prior = (
            F.softplus(self.spectrum_weight) * spectrum_normalized
            + F.softplus(self.mean_alpha_weight) * alpha_mean.squeeze(1)
            + F.softplus(self.alpha_difference_weight) * alpha_difference.squeeze(1)
            + self.coverage_prior_weight * coverage_prior
        )
        spectrum_logits = self.spectrum_gate(spectrum_features).squeeze(1) + prior
        spectrum_gate = torch.sigmoid(spectrum_logits / self.temperature)

        embeddings = self.spectrum_embedding(spectrum_features)
        embeddings = embeddings * spectrum_gate.unsqueeze(1)
        embeddings = embeddings.reshape(batch_size, self.spectrum_embedding_dim, -1)
        gather_indices = flat_indices.unsqueeze(1).expand(
            -1,
            self.spectrum_embedding_dim,
            -1,
        )
        pixel_context = torch.gather(embeddings, 2, gather_indices).reshape(
            batch_size,
            self.spectrum_embedding_dim,
            height,
            width,
        )

        holder_hh_normalized = self._normalize_pixels(holder_hh).unsqueeze(1)
        holder_vv_normalized = self._normalize_pixels(holder_vv).unsqueeze(1)
        holder_features = torch.cat(
            [
                holder_hh_normalized,
                holder_vv_normalized,
                torch.abs(holder_hh_normalized - holder_vv_normalized),
                0.5 * (holder_hh_normalized + holder_vv_normalized),
            ],
            dim=1,
        )
        pixel_features = torch.cat([pixel_context, holder_features], dim=1)
        hh_gate = self._smooth(self._center_gate(torch.sigmoid(self.hh_gate(pixel_features))))
        vv_gate = self._smooth(self._center_gate(torch.sigmoid(self.vv_gate(pixel_features))))
        return hh_gate, vv_gate, spectrum_gate


class ResNet18Encoder(nn.Module):
    """ResNet-18 feature encoder adapted to an arbitrary input channel count."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        if in_channels != 3:
            original = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                in_channels,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                bias=False,
            )
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(inputs)).flatten(1)


class ClassificationHead(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class MVSPSNet(nn.Module):
    """Dual-stream ResNet-18 with MVSPS-guided input filtering.

    The fractal statistics are intentionally computed without gradients using
    NumPy, SciPy, and PyWavelets. Learnable spectrum-to-spatial projection and
    both ResNet streams remain differentiable.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 40,
        num_bins: int = 10,
        dropout: float = 0.2,
        wavelet: str = "db3",
        wavelet_levels: int | None = None,
        leader_neighborhood: int = 3,
        normalize_spectrum_by_count: bool = True,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.in_channels = in_channels
        self.num_bins = num_bins
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.leader_neighborhood = leader_neighborhood
        self.normalize_spectrum_by_count = normalize_spectrum_by_count

        self.mvsps = MVSPSBlock(num_bins=num_bins)
        self.hh_encoder = ResNet18Encoder(in_channels)
        self.vv_encoder = ResNet18Encoder(in_channels)
        self.hh_head = ClassificationHead(num_classes, dropout)
        self.vv_head = ClassificationHead(num_classes, dropout)
        self.hh_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.vv_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.fusion_logits = nn.Parameter(torch.tensor([0.5, 0.5]))

    def _statistics_batch(
        self,
        image_hh: torch.Tensor,
        image_vv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = image_hh.device
        spectra = []
        index_maps = []
        holders_hh = []
        holders_vv = []
        for hh_sample, vv_sample in zip(image_hh, image_vv):
            hh_gray = hh_sample.detach().float().mean(dim=0).cpu().numpy()
            vv_gray = vv_sample.detach().float().mean(dim=0).cpu().numpy()
            spectrum, index_map, holder_hh, holder_vv = (
                multivariate_singularity_power_spectrum(
                    hh_gray,
                    vv_gray,
                    num_bins=self.num_bins,
                    normalize_by_count=self.normalize_spectrum_by_count,
                    wavelet=self.wavelet,
                    levels=self.wavelet_levels,
                    neighborhood=self.leader_neighborhood,
                )
            )
            spectra.append(spectrum.astype(np.float32))
            index_maps.append(index_map.astype(np.int64))
            holders_hh.append(holder_hh.astype(np.float32))
            holders_vv.append(holder_vv.astype(np.float32))

        return (
            torch.from_numpy(np.stack(spectra)).to(device),
            torch.from_numpy(np.stack(index_maps)).to(device),
            torch.from_numpy(np.stack(holders_hh)).to(device),
            torch.from_numpy(np.stack(holders_vv)).to(device),
        )

    def forward(self, image_hh: torch.Tensor, image_vv: torch.Tensor) -> dict[str, torch.Tensor]:
        if image_hh.shape != image_vv.shape or image_hh.ndim != 4:
            raise ValueError(
                "Expected two tensors with identical [B, C, H, W] shapes; "
                f"got {tuple(image_hh.shape)} and {tuple(image_vv.shape)}"
            )
        if image_hh.shape[1] != self.in_channels:
            raise ValueError(
                f"Model expects {self.in_channels} input channels, got {image_hh.shape[1]}"
            )

        with torch.no_grad():
            spectrum, index_map, holder_hh, holder_vv = self._statistics_batch(
                image_hh,
                image_vv,
            )
        spatial_hh, spatial_vv, spectrum_gate = self.mvsps(
            spectrum,
            index_map,
            holder_hh,
            holder_vv,
        )

        mix_hh = torch.sigmoid(self.hh_mix_logit)
        mix_vv = torch.sigmoid(self.vv_mix_logit)
        filtered_hh = image_hh * spatial_hh
        filtered_vv = image_vv * spatial_vv
        input_hh = torch.lerp(image_hh, filtered_hh, mix_hh)
        input_vv = torch.lerp(image_vv, filtered_vv, mix_vv)

        features_hh = self.hh_encoder(input_hh)
        features_vv = self.vv_encoder(input_vv)
        logits_hh = self.hh_head(features_hh)
        logits_vv = self.vv_head(features_vv)
        fusion_weights = torch.softmax(self.fusion_logits, dim=0)
        logits = fusion_weights[0] * logits_hh + fusion_weights[1] * logits_vv

        return {
            "logits": logits,
            "logits_hh": logits_hh,
            "logits_vv": logits_vv,
            "probs": torch.softmax(logits, dim=1),
            "spatial_gate_hh": spatial_hh,
            "spatial_gate_vv": spatial_vv,
            "spectrum_gate": spectrum_gate,
            "spectrum": spectrum,
        }
