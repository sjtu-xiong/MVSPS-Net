"""Paired-image dataset helpers used by the training entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


DEFAULT_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


class PairedPolarizationDataset(Dataset):
    """Load two polarization images that share a class and file identifier.

    The directory layout follows ``ImageFolder``::

        root/
          class_a/
            scene_001_HH_image.tif
            scene_001_VV_image.tif

    A pair is formed by replacing ``hh_token`` with ``vv_token`` in the HH
    filename. The returned sample is ``(image_hh, image_vv, class_index)``.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        *,
        hh_token: str = "_HH_",
        vv_token: str = "_VV_",
        extensions: Iterable[str] = DEFAULT_EXTENSIONS,
        class_to_idx: Mapping[str, int] | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        if not hh_token or not vv_token or hh_token == vv_token:
            raise ValueError("hh_token and vv_token must be different non-empty strings")

        self.transform = transform
        self.hh_token = hh_token
        self.vv_token = vv_token
        self.extensions = {suffix.lower() for suffix in extensions}

        if class_to_idx is None:
            class_names = sorted(path.name for path in self.root.iterdir() if path.is_dir())
            if not class_names:
                raise RuntimeError(f"No class directories found under {self.root}")
            self.class_to_idx = {name: index for index, name in enumerate(class_names)}
        else:
            self.class_to_idx = dict(class_to_idx)
            unknown = sorted(
                path.name
                for path in self.root.iterdir()
                if path.is_dir() and path.name not in self.class_to_idx
            )
            if unknown:
                raise ValueError(
                    "Found classes that are absent from the training split: "
                    + ", ".join(unknown)
                )

        self.classes = [
            name for name, _ in sorted(self.class_to_idx.items(), key=lambda item: item[1])
        ]

        pairs: list[tuple[Path, Path, int]] = []
        unmatched = 0
        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            target = self.class_to_idx[class_name]
            for hh_path in sorted(class_dir.iterdir()):
                if (
                    not hh_path.is_file()
                    or hh_path.suffix.lower() not in self.extensions
                    or self.hh_token not in hh_path.name
                ):
                    continue
                vv_path = hh_path.with_name(
                    hh_path.name.replace(self.hh_token, self.vv_token, 1)
                )
                if vv_path.is_file():
                    pairs.append((hh_path, vv_path, target))
                else:
                    unmatched += 1

        if not pairs:
            raise RuntimeError(
                f"No paired images found under {self.root}. Expected filenames containing "
                f"'{self.hh_token}' with matching '{self.vv_token}' files."
            )

        self.samples = pairs
        if verbose:
            print(
                f"[data] {self.root}: {len(self.samples)} pairs, "
                f"{len(self.class_to_idx)} classes, {unmatched} unmatched HH files"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        hh_path, vv_path, target = self.samples[index]
        image_hh = default_loader(str(hh_path))
        image_vv = default_loader(str(vv_path))
        if self.transform is not None:
            image_hh = self.transform(image_hh)
            image_vv = self.transform(image_vv)
        return image_hh, image_vv, target


def load_paired_data(
    root: str | Path,
    transform: Callable | None = None,
    **kwargs,
) -> PairedPolarizationDataset:
    """Convenience wrapper matching the original notebook API."""

    return PairedPolarizationDataset(root, transform, **kwargs)
