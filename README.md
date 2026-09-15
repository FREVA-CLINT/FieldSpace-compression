# FieldSpaceNN compression notebooks

This repository provides a notebook-based workflow for training and evaluating
neural compression models for global climate fields. The models are supplied by
[FieldSpaceNN](https://github.com/FREVA-CLINT/FieldSpaceNN), and operate on
multiresolution fields represented on a nested HEALPix grid.

The repository covers the complete experiment lifecycle:

1. remap a NetCDF field from a regular, curvilinear, or unstructured grid to
   HEALPix;
2. train a multigrid hyperprior autoencoder in two stages; and
3. entropy-code data with a trained checkpoint, decode it again, and measure the
   resulting compression and reconstruction quality.

The notebooks contain the experiment controls and narrative. Shared validation,
plotting, serialization, and configuration code lives in `notebooks/utils.py`.
See [the notebook guide](notebooks/README.md) for a detailed walkthrough of all
three notebooks.

## Repository layout

```text
.
├── README.md
├── requirements.txt
└── notebooks/
    ├── README.md
    ├── base_config.yaml
    ├── utils.py
    ├── remap_any_to_healpix.ipynb
    ├── fieldspace_training.ipynb
    └── fieldspace_inference.ipynb
```

`base_config.yaml` is the HPX5/7/9 model template. The training notebook derives
a fully resolved experiment configuration from it; it should normally not be
edited directly.

## Prerequisites

- Git; a 64-bit Linux environment is recommended and matches the intended
  cluster workflow
- Python 3.11 or 3.12 (Python 3.11 is recommended)
- enough local or project storage for the source fields, checkpoints, compressed
  shards, and decoded Zarr output
- a CUDA-capable GPU for practical high-resolution training; remapping and small
  inference experiments can run on CPU
- an existing JupyterLab/Jupyter Notebook installation, or permission to install
  one

The default paths in the notebooks point to project filesystems used by the
authors. They are examples, not bundled data. Replace them with paths that are
available on your system.

## Installation

Clone this repository and create an isolated environment:

```bash
git clone https://github.com/FREVA-CLINT/FieldSpace-compression.git
cd FieldSpace-compression

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

PyTorch 2.4.0 is pinned because the model stack was prepared and tested with
that release. Installing the desired PyTorch build first prevents `pip` from
choosing an unsuitable accelerator build.

For CPU-only use:

```bash
python -m pip install \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For an NVIDIA system compatible with the CUDA 12.1 PyTorch wheels:

```bash
python -m pip install \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The requirements install FieldSpaceNN directly from its `dev-jm` development
branch. They list only the packages used directly by this repository; secondary
packages are resolved by `pip` from those dependencies. Because a branch can
move, reinstalling at a later date may retrieve newer FieldSpaceNN code.

Register the environment as a notebook kernel:

```bash
python -m ipykernel install --user \
  --name fieldspace-compression \
  --display-name "Python (FieldSpace compression)"
```

If the machine does not already provide a Jupyter frontend, install one and
start it from the repository root:

```bash
python -m pip install jupyterlab
jupyter lab
```

Select **Python (FieldSpace compression)** as the kernel. Starting Jupyter from
either the repository root or `notebooks/` is supported.

### Verify the environment

Run this import check before opening a long training job:

```bash
python -c "import compressai, dask, healpy, hydra, lightning, matplotlib, netCDF4, numpy, pandas, scipy, s3fs, torch, xarray, zarr; import fieldspacenn; print('environment OK'); print('CUDA:', torch.cuda.is_available())"
```

If `CUDA: False` is printed on a GPU node, confirm that the selected PyTorch
wheel matches the installed NVIDIA driver and that the notebook kernel uses the
new virtual environment.

## Input data

Training and inference expect the highest requested HEALPix resolution in a
Zarr store. The lower two resolutions are generated lazily by the FieldSpaceNN
loader. Each selected field must have:

- a `time` dimension;
- one spatial dimension named `cell` or `ncells`;
- `12 * 4**zoom` spatial values for HEALPix level `zoom`; and
- at most one additional vertical dimension.

Variables may be split into two groups. A `2D` variable must resolve to exactly
one vertical level (or have no vertical dimension). A `3D` variable may retain
all levels or select a list of levels, but every variable in that group must
resolve to the same depth. When variables are read from different stores, their
time and cell coordinates must align.

If the source is NetCDF on another grid, run
`notebooks/remap_any_to_healpix.ipynb` first. Its default output uses nested
HEALPix ordering, which matches the compression notebooks. The remapper performs
interpolation at cell centers; it is not conservative and therefore should not
be used unchanged when exact preservation of an area integral is required.

Remote Zarr URLs are supported through `s3fs`. Authentication, endpoint
configuration, and access rights remain specific to the archive being used.

## Running the workflow

### 1. Prepare HEALPix data

Open `notebooks/remap_any_to_healpix.ipynb` and edit its configuration cell:

- `INPUT_PATH` and `OUTPUT_PATH`
- `HEALPIX_LEVEL`
- `VARIABLES` and optional timestep/level selections
- coordinate-name overrides, if CF metadata is insufficient
- `OVERWRITE_OUTPUT`

Run the notebook top to bottom and inspect the Mollweide preview before using the
written Zarr store.

### 2. Train a compressor

Open `notebooks/fieldspace_training.ipynb` and edit the single experiment-control
cell. At minimum, set the Zarr path for each variable and choose unique
`PROJECT_NAME` and `RUN_NAME` values. For a quick smoke test, the notebook
suggests:

```python
IN_ZOOMS = [1, 3, 5]
MODEL_COMPLEXITY = 0.125
MAX_STEPS = 4
```

Use an appropriately larger model and step budget for a scientific experiment.
`MAX_STEPS` applies to each of the two stages. Training reserves the last 20
selected timesteps: ten for validation and ten for testing. Consequently, at
least 21 timesteps must be available.

The run produces, among other files:

```text
notebooks/snapshots/<PROJECT_NAME>/<RUN_NAME>/
├── composed_config.yaml
├── normalization_mean_std.json
├── last.ckpt
├── input_zoom_preview.png
├── input_zoom_decomposition.png
├── training_loss.png
└── csv_logs/

notebooks/snapshots/<PROJECT_NAME>/<RUN_NAME>_finetune/
├── composed_config.yaml
├── last.ckpt
└── csv_logs/
```

The final cell prints a complete model-profile entry for the inference notebook.
Keep each `last.ckpt` together with its generated `composed_config.yaml`: model
dimensions, variables, normalization, and HEALPix levels are coupled.

### 3. Compress and evaluate

Open `notebooks/fieldspace_inference.ipynb`, paste the generated profile into
`MODEL_PROFILES`, and select it with `MODEL_PROFILE`. Check the checkpoint,
configuration, normalization, and data paths before running all cells.

Inference writes actual entropy-coded `y` and `z` streams in sharded `.pt`
artifacts, reloads those artifacts for a genuine encode/decode round trip, and
reports compression ratio, bits per value, timing, RMSE, MAE, bias, and Pearson
correlation. It also produces reconstruction maps and can export denormalized
decoded fields to `decoded_fields.zarr`.

By default, inference output is written below:

```text
notebooks/evaluations/hackathon_hyperprior/<MODEL_PROFILE>/
```

The exact output bundle and notebook controls are documented in
[notebooks/README.md](notebooks/README.md).

## Resource and cluster notes

- HEALPix grows by a factor of four per level: `Npix = 12 * 4**zoom`. HPX9 has
  3,145,728 cells per global map.
- Lower `BATCH_SIZE` first when training runs out of GPU memory.
- `NUM_WORKERS = 0` is the safest choice for interactive inference. Training can
  use more workers when the compute environment supports multiprocessing.
- Exact normalization reads every finite training value and can be I/O-heavy.
- The inference path processes and shards samples incrementally, and the decoded
  Zarr exporter also streams records, limiting peak memory for longer runs.
- Plotting caches are redirected to `/tmp` by the notebooks to work on clusters
  with read-only or restricted home directories.

## Troubleshooting

**A configured file cannot be found.** Replace the example `/p/...` or `/work/...`
paths in the relevant control cell. Inference also needs the normalization JSON
referenced by the saved configuration.

**A Zarr store fails validation.** Confirm the variable names, `time` dimension,
`cell`/`ncells` dimension, selected vertical levels, and expected HEALPix cell
count. Stores used together must describe aligned samples.

**The process runs out of memory.** Reduce the training batch size, use the lower
resolution smoke-test configuration, reduce `MODEL_COMPLEXITY` in increments
that retain whole 32-channel attention heads, or select a larger GPU. CPU
inference is supported but high-resolution runs can be slow.

**The checkpoint does not load.** Use the `composed_config.yaml` generated by the
same fine-tuning run. Variable order, 2D/3D grouping, selected depth, model
complexity, and zoom levels affect checkpoint tensor shapes.

**A remote store does not open.** Verify the URL and credentials, and test that
the active environment can import `s3fs`. Some archives require site-specific
authentication or network access.
