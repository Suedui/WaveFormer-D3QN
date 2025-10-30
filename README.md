# WaveFormer-D3QN

WaveFormer-D3QN is a reference implementation of a Transformer-based model that
embeds a discrete wavelet transform in its feature extractor. A Dueling Double
Deep Q-Network (D3QN) agent adaptively selects the most suitable wavelet kernel
for each batch, reducing the reliance on manual expert tuning.

## Project structure

```
waveformer/
├── __init__.py
├── dataset.py          # NumPy-backed dataset loader with configurable root path
├── d3qn.py             # D3QN agent components for wavelet kernel selection
├── model.py            # WaveFormer architecture definition
└── wavelet_transform.py# Wavelet utilities built on top of PyWavelets

train.py                # End-to-end training script
```

## Requirements

Install the required Python packages (PyTorch and PyWavelets) before running
the project:

```bash
pip install torch pywavelets
```

## Preparing the dataset

Place your dataset under a directory of your choice (default: `data/your-dataset`).
The loader expects two NumPy arrays:

- `signals.npy`: shape `(num_samples, sequence_length)`
- `targets.npy`: shape `(num_samples,)` for regression or `(num_samples, target_dim)`

You can override the dataset path via the `--dataset-root` argument when
running the training script.

## Training

```bash
python train.py --dataset-root path/to/dataset --epochs 20 --batch-size 64
```

During training the D3QN agent observes coarse statistics (mean, standard
deviation, sequence length) of each batch to select a wavelet kernel. The
resulting wavelet coefficients are projected into the Transformer encoder for
downstream prediction.

## Extending

- Adjust the candidate wavelet kernels in `WaveFormerConfig.wavelet_kernels`.
- Modify reward shaping logic in `train.py::compute_reward` to better match
  your task.
- Swap in a custom Transformer head in `waveformer/model.py` as needed.

