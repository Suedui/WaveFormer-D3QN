"""Command line interface for training the WaveFormer-D3QN model on the CWRU dataset."""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, _LRScheduler
from torch.utils.data import DataLoader, Subset, random_split

from utils.train_utils import (
    _prepare_class_labels,
    evaluate,
    evaluate_with_metrics,
    train_epoch,
)

from Datasets import CWRUConfig, CWRUDataset
from Models import WaveFormerConfig, WaveFormerModel
from utils.logger import configure_logging


def _build_labels_tensor(dataset: CWRUDataset) -> torch.Tensor:
    labels = _prepare_class_labels(dataset.targets)
    return labels.cpu()


def _compute_partition_sizes(total: int, ratios: Iterable[float]) -> List[int]:
    raw_sizes = [ratio * total for ratio in ratios]
    base_sizes = [int(math.floor(size)) for size in raw_sizes]
    remainder = total - sum(base_sizes)

    fractional = [size - base for size, base in zip(raw_sizes, base_sizes)]
    order = sorted(range(len(base_sizes)), key=lambda idx: fractional[idx], reverse=True)
    for idx in range(remainder):
        base_sizes[order[idx % len(order)]] += 1

    if total >= len(base_sizes):
        for idx, size in enumerate(base_sizes):
            if size == 0:
                donor = max(range(len(base_sizes)), key=lambda j: base_sizes[j])
                if base_sizes[donor] <= 1:
                    continue
                base_sizes[donor] -= 1
                base_sizes[idx] += 1

    return base_sizes


def _stratified_split_three_way(
    dataset: CWRUDataset,
    ratios: Tuple[float, float, float],
    generator: torch.Generator,
) -> Tuple[Subset[CWRUDataset], Subset[CWRUDataset], Subset[CWRUDataset]]:
    labels = _build_labels_tensor(dataset)
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        class_indices[int(label)].append(idx)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for indices in class_indices.values():
        cls_tensor = torch.tensor(indices, dtype=torch.long)
        if cls_tensor.numel() < 3:
            # Not enough samples for a strict three-way split; fall back to round-robin.
            shuffled = cls_tensor.tolist()
            for pos, sample_idx in enumerate(shuffled):
                if pos % 3 == 0:
                    train_indices.append(sample_idx)
                elif pos % 3 == 1:
                    val_indices.append(sample_idx)
                else:
                    test_indices.append(sample_idx)
            continue

        perm = torch.randperm(cls_tensor.numel(), generator=generator)
        shuffled = cls_tensor[perm].tolist()
        cls_sizes = _compute_partition_sizes(cls_tensor.numel(), ratios)

        cursor = 0
        for size, collection in zip(
            cls_sizes, (train_indices, val_indices, test_indices)
        ):
            if size <= 0:
                continue
            next_cursor = cursor + size
            collection.extend(shuffled[cursor:next_cursor])
            cursor = next_cursor

    if not train_indices or not val_indices or not test_indices:
        total = len(dataset)
        lengths = _compute_partition_sizes(total, ratios)
        subsets = random_split(dataset, lengths, generator=generator)
        return subsets[0], subsets[1], subsets[2]

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def _parse_scale_bands(value: str) -> List[Tuple[float, float]]:
    bands: List[Tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            lower_str, upper_str = item.split("-")
            lower = float(lower_str)
            upper = float(upper_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Scale bands must be provided as 'lower-upper' pairs."
            ) from exc
        if lower <= 0 or upper <= 0:
            raise argparse.ArgumentTypeError("Scale band bounds must be positive numbers.")
        if lower >= upper:
            raise argparse.ArgumentTypeError(
                "Scale band lower bound must be strictly smaller than the upper bound."
            )
        bands.append((lower, upper))

    if not bands:
        raise argparse.ArgumentTypeError("At least one scale band must be provided.")

    return bands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dataset_root = Path(__file__).resolve().parent / "Datasets" / "CWRU"
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_dataset_root,
        help=(
            "Path to the directory containing signals.npy and targets.npy. "
            "Defaults to the bundled CWRU dataset shipped with the project."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay applied by the optimizer for L2 regularization.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for testing.",
    )
    parser.add_argument(
        "--stratified-split",
        action="store_true",
        help="Enable stratified splitting; defaults to random_split when omitted.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable Gaussian noise augmentation for the CWRU dataset.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Standard deviation of the Gaussian noise augmentation.",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=0.0,
        help="Gradient clipping norm applied during training (set >0 to enable).",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "step", "plateau"],
        default="none",
        help="Learning rate scheduler strategy.",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=30,
        help="Step size for StepLR when --lr-scheduler=step.",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.5,
        help="Multiplicative factor of learning rate decay.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="Number of epochs with no improvement before reducing LR when using ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Number of epochs to wait for validation improvement before stopping (0 to disable).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation accuracy improvement required to reset early stopping patience.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate applied inside the WaveFormer transformer backbone.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers in the WaveFormer backbone.",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        default=512,
        help="Hidden dimensionality of the transformer encoder.",
    )
    parser.add_argument(
        "--max-transformer-len",
        type=int,
        default=512,
        help=(
            "Maximum token length fed into the transformer. Sequences exceeding "
            "this length are downsampled to fit the budget."
        ),
    )
    parser.add_argument(
        "--wavelet-num-scales",
        type=int,
        default=64,
        help="Number of scales sampled when computing the wavelet scalogram.",
    )
    parser.add_argument(
        "--wavelet-scale-bands",
        type=_parse_scale_bands,
        default=_parse_scale_bands("1-32,1-64,1-128"),
        help=(
            "Comma-separated scale bands given as 'lower-upper'. "
            "Example: '1-32,1-64,1-128'."
        ),
    )
    parser.add_argument(
        "--sampling-period",
        type=float,
        default=1.0,
        help="Sampling period associated with the input signals for the CWT.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("checkpoint/logs"),
        help="Directory used for training logs.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("checkpoint/waveformer.ckpt"),
        help="Path for saving the final checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    log_dir = args.log_dir.expanduser()
    checkpoint_path = args.checkpoint_path.expanduser()
    device = torch.device(args.device)

    configure_logging(log_dir)

    val_ratio = max(0.0, min(args.val_ratio, 0.9))
    test_ratio = max(0.0, min(args.test_ratio, 0.9))
    if args.val_ratio != val_ratio:
        logging.warning(
            "Validation ratio %.3f adjusted to %.3f to keep it within a safe range.",
            args.val_ratio,
            val_ratio,
        )
    if args.test_ratio != test_ratio:
        logging.warning(
            "Test ratio %.3f adjusted to %.3f to keep it within a safe range.",
            args.test_ratio,
            test_ratio,
        )
    if val_ratio == 0.0:
        logging.warning("Validation ratio of 0.0 detected; defaulting to 0.2 for splitting.")
        val_ratio = 0.2
    if test_ratio == 0.0:
        logging.warning("Test ratio of 0.0 detected; defaulting to 0.2 for splitting.")
        test_ratio = 0.2

    if val_ratio + test_ratio >= 0.99:
        scale = 0.99 / (val_ratio + test_ratio)
        logging.warning(
            "Validation and test ratios sum to %.3f; scaling both by %.3f to retain training samples.",
            val_ratio + test_ratio,
            scale,
        )
        val_ratio *= scale
        test_ratio *= scale

    train_ratio = 1.0 - val_ratio - test_ratio
    ratios = (train_ratio, val_ratio, test_ratio)

    num_scales = max(1, args.wavelet_num_scales)
    if num_scales != args.wavelet_num_scales:
        logging.warning(
            "Wavelet num scales %d adjusted to %d to remain positive.",
            args.wavelet_num_scales,
            num_scales,
        )

    dataset_config = CWRUConfig(
        root=dataset_root,
        augment=args.augment,
        noise_std=args.noise_std,
    )
    dataset = CWRUDataset(dataset_config)
    num_samples = len(dataset)
    if num_samples < 3:
        logging.warning(
            "Dataset contains fewer than three samples; using the same data for all splits.",
        )
        train_dataset = dataset
        val_dataset = dataset
        test_dataset = dataset
    else:
        generator = torch.Generator().manual_seed(42)
        lengths = _compute_partition_sizes(num_samples, ratios)
        if args.stratified_split:
            logging.info(
                "Using stratified split with ratios train=%.2f | val=%.2f | test=%.2f.",
                train_ratio,
                val_ratio,
                test_ratio,
            )
            train_dataset, val_dataset, test_dataset = _stratified_split_three_way(
                dataset,
                ratios,
                generator,
            )
        else:
            logging.info(
                "Using random_split with ratios train=%.2f | val=%.2f | test=%.2f.",
                train_ratio,
                val_ratio,
                test_ratio,
            )
            subsets = random_split(dataset, lengths, generator=generator)
            train_dataset, val_dataset, test_dataset = subsets

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    logging.info(
        "Dataset split: %d training samples | %d validation samples | %d test samples",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    sample_signal = dataset[0][0].numpy()
    signal_length = sample_signal.shape[-1]

    max_transformer_len = max(1, args.max_transformer_len)
    if max_transformer_len != args.max_transformer_len:
        logging.warning(
            "Maximum transformer length %d adjusted to %d to remain positive.",
            args.max_transformer_len,
            max_transformer_len,
        )
    target_seq_len = min(signal_length, max_transformer_len)
    if target_seq_len < signal_length:
        logging.info(
            "Downsampling scalograms from %d to %d time steps before transformer encoding.",
            signal_length,
            target_seq_len,
        )

    scale_bands = tuple(tuple(band) for band in args.wavelet_scale_bands)

    labels = _build_labels_tensor(dataset)
    num_classes = int(labels.max().item()) + 1
    model_config = WaveFormerConfig(
        input_dim=num_scales,
        model_dim=args.model_dim,
        output_dim=num_classes,
        max_seq_len=target_seq_len,
        wavelet_scale_bands=scale_bands,
        wavelet_num_scales=num_scales,
        sampling_period=args.sampling_period,
        dropout=args.dropout,
        num_layers=args.num_layers,
    )
    model = WaveFormerModel(model_config, device=device).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler: _LRScheduler | ReduceLROnPlateau | None
    if args.lr_scheduler == "step":
        scheduler = StepLR(
            optimizer,
            step_size=max(1, args.lr_step_size),
            gamma=args.lr_gamma,
        )
    elif args.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_gamma,
            patience=max(1, args.lr_patience),
            verbose=True,
        )
    else:
        scheduler = None

    best_train_acc = 0.0
    best_train_epoch = 0
    best_val_acc = 0.0
    best_val_epoch = 0
    final_train_acc = 0.0
    final_val_acc = 0.0
    best_model_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    patience = max(0, args.early_stopping_patience)
    min_delta = args.early_stopping_min_delta
    gradient_clip = None if args.gradient_clip <= 0 else args.gradient_clip

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gradient_clip=gradient_clip,
        )
        val_loss, val_acc = evaluate(model, val_loader, device)

        if train_acc >= best_train_acc:
            best_train_acc = train_acc
            best_train_epoch = epoch
        if val_acc >= best_val_acc + min_delta:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        final_train_acc = train_acc
        final_val_acc = val_acc

        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(
            (
                "Epoch %03d | Train Loss: %.6f | Train Acc: %.4f | "
                "Val Loss: %.6f | Val Acc: %.4f | LR: %.6f"
            ),
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            current_lr,
        )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if patience and patience_counter >= patience:
            logging.info(
                "Early stopping triggered after %d epochs without validation improvement.",
                patience_counter,
            )
            break

    logging.info(
        "Max train acc: %.4f (epoch %03d) | Max val acc: %.4f (epoch %03d)",
        best_train_acc,
        best_train_epoch,
        best_val_acc,
        best_val_epoch,
    )
    logging.info(
        "Final train acc: %.4f | Final val acc: %.4f",
        final_train_acc,
        final_val_acc,
    )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Restored model weights from epoch %03d for checkpointing.", best_val_epoch)

    test_metrics = evaluate_with_metrics(model, test_loader, device, num_classes)
    logging.info(
        "Test metrics | Accuracy: %.4f | Recall: %.4f | F1-score: %.4f",
        test_metrics["accuracy"],
        test_metrics["recall"],
        test_metrics["f1"],
    )

    logging.info("Training complete. Saving checkpoint to %s", checkpoint_path.resolve())
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    main()
