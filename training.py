"""Training and evaluation loops for dual-input classification models."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def extract_logits(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, Mapping) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    if isinstance(output, (list, tuple)) and output:
        return extract_logits(output[0])
    raise TypeError(f"Unsupported model output: {type(output)!r}")


def classification_loss(
    output,
    targets: torch.Tensor,
    *,
    auxiliary_weight: float = 0.3,
    consistency_weight: float = 0.05,
    consistency_temperature: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = extract_logits(output)
    main_loss = F.cross_entropy(logits, targets)
    auxiliary_loss = logits.new_zeros(())
    consistency_loss = logits.new_zeros(())

    logits_hh = output.get("logits_hh") if isinstance(output, Mapping) else None
    logits_vv = output.get("logits_vv") if isinstance(output, Mapping) else None
    has_auxiliary_heads = isinstance(logits_hh, torch.Tensor) and isinstance(
        logits_vv,
        torch.Tensor,
    )
    if has_auxiliary_heads:
        auxiliary_loss = F.cross_entropy(logits_hh, targets) + F.cross_entropy(
            logits_vv,
            targets,
        )
        if consistency_weight > 0:
            temperature = consistency_temperature
            main_log_probabilities = F.log_softmax(logits / temperature, dim=1)
            mean_branch_probabilities = 0.5 * (
                F.softmax(logits_hh / temperature, dim=1)
                + F.softmax(logits_vv / temperature, dim=1)
            )
            consistency_loss = temperature**2 * F.kl_div(
                main_log_probabilities,
                mean_branch_probabilities.detach(),
                reduction="batchmean",
            )

    total_loss = (
        main_loss
        + auxiliary_weight * auxiliary_loss
        + consistency_weight * consistency_loss
    )
    metrics = {
        "loss": float(total_loss.detach()),
        "main_loss": float(main_loss.detach()),
        "auxiliary_loss": float(auxiliary_loss.detach()),
        "consistency_loss": float(consistency_loss.detach()),
    }
    return total_loss, metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    scaler=None,
    use_amp: bool = False,
    max_grad_norm: float | None = 1.0,
    auxiliary_weight: float = 0.3,
    consistency_weight: float = 0.05,
    consistency_temperature: float = 2.0,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    sample_count = 0
    amp_enabled = use_amp and device.type == "cuda"
    # The MVSPS branch contains CPU/NumPy statistics and small spectrum tensors.
    # float16 offers the broadest CUDA compatibility for the differentiable path.
    amp_dtype = torch.float16

    progress = tqdm(loader, desc="train", leave=False)
    for image_hh, image_vv, targets in progress:
        image_hh = image_hh.to(device, non_blocking=True)
        image_vv = image_vv.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_enabled
            else nullcontext()
        )
        with amp_context:
            output = model(image_hh, image_vv)
            loss, _ = classification_loss(
                output,
                targets,
                auxiliary_weight=auxiliary_weight,
                consistency_weight=consistency_weight,
                consistency_temperature=consistency_temperature,
            )

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        logits = extract_logits(output)
        batch_size = targets.size(0)
        running_loss += float(loss.detach()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum())
        sample_count += batch_size
        progress.set_postfix(
            loss=f"{running_loss / sample_count:.4f}",
            acc=f"{100.0 * correct / sample_count:.2f}%",
        )

    return {
        "loss": running_loss / max(1, sample_count),
        "accuracy": 100.0 * correct / max(1, sample_count),
    }


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    sample_count = 0
    progress = tqdm(loader, desc="validate", leave=False)
    for image_hh, image_vv, targets in progress:
        image_hh = image_hh.to(device, non_blocking=True)
        image_vv = image_vv.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = extract_logits(model(image_hh, image_vv))
        loss = F.cross_entropy(logits, targets)
        batch_size = targets.size(0)
        running_loss += float(loss) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum())
        sample_count += batch_size
        progress.set_postfix(acc=f"{100.0 * correct / sample_count:.2f}%")

    return {
        "loss": running_loss / max(1, sample_count),
        "accuracy": 100.0 * correct / max(1, sample_count),
    }
