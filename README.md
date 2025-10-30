# WaveFormer-D3QN

WaveFormer-D3QN is a reference implementation of a transformer-enhanced fault diagnosis
pipeline. A discrete wavelet transform feeds a transformer encoder, while a
Dueling Double Deep Q-Network (D3QN) agent adaptively selects the most suitable
wavelet kernel for each batch, reducing the reliance on manual expert tuning.

## Project structure

```
WaveFormer-D3QN/
├── main.py                 # Command line entry point
├── train.py                # Backwards-compatible wrapper around main.py
├── Datasets/               # Dataset definitions and augmentation helpers
│   ├── CWRU.py
│   └── Dataset_utils/
│       ├── DatasetsBase.py
│       ├── sequence_aug.py
│       └── get_files/
│           ├── CWRU_get_files.py
│           └── generalfunction.py
├── Models/                 # Model components
│   ├── BackboneTransformer.py # Transformer backbone
│   ├── WaveFormer.py       # High-level WaveFormer model tying everything together
│   ├── TFconvlayer.py      # Wavelet transform utilities
│   └── WaveletRLConv.py    # D3QN agent for kernel selection
├── PostProcess/            # Evaluation and visualisation helpers
│   ├── Acc_statistic.py
│   ├── fg_cam.py
│   ├── TrainSequentially.py
│   └── process_utils/
│       ├── PlotAccuracy.py
│       └── processlib.py
├── utils/                  # Generic utilities
│   ├── logger.py
│   ├── mysummary.py
│   └── train_utils.py
├── checkpoint/             # Default location for logs and checkpoints
└── Doc/                    # Documentation and figures
```

## Requirements

Install the required Python packages (PyTorch, PyWavelets, Matplotlib, NumPy,
and optional TorchInfo for summaries) before running the project:

```bash
pip install torch pywavelets matplotlib numpy torchinfo
```

## Preparing the dataset

Place your dataset under a directory of your choice (default: `C:/Users/刘明浩/PycharmProjects/WaveFormer-D3QN/Datasets/CWRU`).
The loader now accepts either the original CWRU `.mat` files or pre-converted
NumPy arrays. When `.mat` files are detected they are converted on-the-fly and
cached as:

- `signals.npy`: shape `(num_samples, sequence_length)`
- `targets.npy`: shape `(num_samples,)` for regression or `(num_samples, target_dim)`

You can override the dataset path via the `--dataset-root` argument when running
`main.py`.

## Training

```bash
python main.py --dataset-root path/to/dataset --epochs 20 --batch-size 64
```

During training the D3QN agent observes coarse statistics (mean, standard
deviation, sequence length) of each batch to select a wavelet kernel. The
resulting wavelet coefficients are projected into the Transformer encoder for
downstream prediction.

## Post-processing

- Use `PostProcess/Acc_statistic.py` to aggregate accuracy statistics.
- Generate gradient-based activation maps with `PostProcess/fg_cam.py`.
- Plot accuracy curves using `PostProcess/process_utils/PlotAccuracy.py`.

## Extending

- Adjust the candidate wavelet kernels in `Models/WaveFormer.WaveFormerConfig.wavelet_kernels`.
- Modify reward shaping logic in `utils/train_utils.compute_reward` to better
  match your task.
- Swap in a custom Transformer head in `Models/BackboneTransformer.py` as needed.
