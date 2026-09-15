# Notebook guide

The three notebooks in this directory form a complete preparation, training,
and evaluation workflow for FieldSpaceNN hyperprior compression. Each notebook
is designed to be executed from top to bottom and keeps its user-editable
settings near the beginning.

Use them in this order when starting from non-HEALPix data:

```text
NetCDF climate fields
        │
        ▼
remap_any_to_healpix.ipynb
        │  nested HEALPix Zarr
        ▼
fieldspace_training.ipynb
        │  resolved config + fine-tuned checkpoint
        ▼
fieldspace_inference.ipynb
        │  entropy-coded shards + metrics + decoded Zarr
        ▼
scientific analysis / archival evaluation
```

If suitable HEALPix Zarr data and a compatible trained checkpoint already
exist, start with training or inference as appropriate.

## Shared files

`utils.py` contains the reusable implementation used by training and inference:
configuration validation, timestep splitting, exact normalization, data
inspection, artifact packing, streaming metrics, plotting, and decoded-Zarr
export. Both notebooks locate this file whether Jupyter was launched in the
repository root or in this directory.

`base_config.yaml` is an HPX5/7/9 architecture template. The training notebook
maps its low/medium/high levels to the requested `IN_ZOOMS`, updates all coupled
model dimensions and data definitions, and writes the resolved result into the
run directory. Make experiment changes in the notebook controls instead of
editing isolated values in this template.

The absolute data and checkpoint paths currently shown in the notebooks are
site-specific examples. Replace them before running the corresponding workflow.

## `remap_any_to_healpix.ipynb`

### Purpose

This notebook converts spatial fields in a NetCDF file to a nested HEALPix Zarr
store suitable for the other two notebooks. It supports:

- rectilinear latitude-longitude grids;
- curvilinear grids with shared 2-D longitude and latitude coordinates; and
- unstructured grids, including ICON-style cell-center coordinates.

It selects variables, timesteps, and optional vertical levels without requiring
the source to use a particular variable naming convention.

### Controls to edit

- `INPUT_PATH`: source NetCDF file.
- `HEALPIX_LEVEL`: target HEALPix level. The output contains
  `12 * 4**HEALPIX_LEVEL` cells.
- `OUTPUT_PATH`: destination Zarr store. It can also be set with the
  `HEALPIX_REMAP_OUTPUT` environment variable.
- `VARIABLES`: optional `2D`/`3D` variable mapping. A `timesteps` entry accepts
  individual indices and end-exclusive ranges such as `"5-100"`. With no field
  entries, all numeric variables on the detected horizontal grid are remapped.
- `LATITUDE_NAME`, `LONGITUDE_NAME`, and `TIME_DIMENSION`: optional overrides for
  unconventional source metadata.
- `LEVEL_DIMENSIONS`: per-variable vertical-dimension overrides.
- `HEALPIX_NESTED`: keep this `True` for direct use by the compression workflow.
- `OUTPUT_DTYPE`: `float32` or `float64`.
- `OVERWRITE_OUTPUT`: whether an existing destination may be replaced.

### What it does, section by section

1. It creates writable plotting/cache directories under `/tmp`, imports the
   scientific stack, and reports key package versions.
2. It validates and expands timestep and level selections. Coordinate detection
   scores CF `standard_name`, `axis`, units, and common coordinate names.
3. It converts angular coordinates to degrees when necessary and constructs
   HEALPix cell centers in nested or ring ordering.
4. For a rectilinear grid, it sorts coordinates, removes a duplicate 0/360-degree
   longitude, adds periodic edge columns, and performs paired vectorized linear
   interpolation with a nearest fallback.
5. For a curvilinear or unstructured grid, it builds a seam-aware Delaunay
   triangulation in longitude/latitude, linearly interpolates every independent
   time/level slice, and fills unresolved targets with a spherical nearest
   neighbour.
6. It displays the detected source geometry and a table containing the method
   and selected shape for every output variable.
7. It plots the first selected field as a HEALPix Mollweide map. Use this to spot
   transposed coordinates, unit mistakes, dateline seams, and missing regions.
8. It chunks and writes a consolidated, Zarr-format-2 store, reopens it, and
   checks variables, dimensions, HEALPix metadata, and finite values.

The output retains dataset, variable, time, and level metadata and adds `cell`,
`lon`, `lat`, HEALPix grid-mapping information, source provenance, and a history
entry.

### Important limitation

The notebook interpolates values at HEALPix cell centers. It does not calculate
source/target area-overlap weights and is therefore not conservative. Replace
the interpolation engine with a conservative remapper when exact preservation
of area-integrated quantities, such as accumulated precipitation, is required.

## `fieldspace_training.ipynb`

### Purpose

This notebook trains a multigrid hyperprior autoencoder in two sequential stages:

1. **pretraining** trains the analysis transform, synthesis transform, and an
   initial entropy model for multiscale reconstruction;
2. **joint fine-tuning** initializes from the pretraining checkpoint, freezes the
   analysis encoder, and optimizes the decoder and entropy model for the final
   rate-distortion objective.

Each stage runs for `MAX_STEPS`; a complete run therefore executes two training
stages with that budget. Logging is local through Lightning's CSV logger and no
Weights & Biases account is needed.

### Controls to edit

- `MAX_STEPS`: optimizer steps per stage.
- `BATCH_SIZE`: timesteps per optimizer step. It has a large memory impact at
  high HEALPix levels.
- `IN_ZOOMS`: exactly three distinct levels, interpreted in sorted
  low/medium/high order. The source Zarr data must be at the highest level.
- `VARIABLES`: one or both of the `2D` and `3D` groups, with a `path` for every
  field and optional `level_indices`. An optional `timesteps` entry limits the
  sample pool.
- `MODEL_COMPLEXITY`: scales the base 256-channel attention width. The result
  must be divisible by the 32-channel head width.
- `PROJECT_NAME` and `RUN_NAME`: determine snapshot and log locations.
- `OVERWRITE_EXISTING_RUN`: permits deletion of known notebook-generated files
  in the same run directories. Choose a new run name when retaining an existing
  experiment.
- `NUM_WORKERS`: data-loading worker processes.
- `OUTPUT_ROOT`: parent of the generated `snapshots/` tree.

For a 2-D group, each variable must have no vertical dimension or select exactly
one level. A 3-D variable may keep all source levels or select any list, but all
3-D variables must have the same selected depth. Different variable stores must
agree in time and cell geometry.

### What it does, section by section

1. It configures caches, random seeds, and the available accelerator.
2. It validates every control and input store before allocating the model. The
   HEALPix level is derived from the cell count and compared with the largest
   configured zoom.
3. It creates deterministic splits. The last ten selected samples are the test
   set, the preceding ten are validation, and all earlier samples are training.
   At least 21 selected timesteps are required.
4. It streams the complete training split to calculate population mean and
   standard deviation. Two-dimensional fields receive scalar statistics;
   three-dimensional fields receive per-source-level statistics. Validation and
   test data never contribute to normalization.
5. It derives a self-contained pretraining configuration from `base_config.yaml`.
   This updates zoom mappings, variable groups, depth tokens, attention widths,
   embeddings, loss levels, sampler settings, file paths, splits, logging, and
   checkpoint intervals together.
6. It instantiates the exact train/validation datasets and plots both composed
   multiresolution fields and the coarse-plus-residual decomposition seen by the
   model.
7. It runs pretraining and saves periodic checkpoints plus `last.ckpt`.
8. It copies the resolved configuration for fine-tuning, changes the stage to
   `joint`, freezes the analysis encoder, references the pretraining checkpoint,
   and trains the second stage.
9. It plots train/validation loss from both CSV logs and the estimated
   fine-tuning compression ratio. It reports best/final validation values.
10. It prints a complete `MODEL_PROFILES` entry containing the fine-tuned
    checkpoint, fine-tuning config, input paths, levels, zooms, and test selection
    for use in inference.

### Outputs

Pretraining is written to:

```text
<OUTPUT_ROOT>/snapshots/<PROJECT_NAME>/<RUN_NAME>/
```

The directory contains `normalization_mean_std.json`, `composed_config.yaml`,
`last.ckpt`, periodic checkpoints, local CSV metrics, data previews, and the
combined `training_loss.png` generated after both stages.

Fine-tuning is written to:

```text
<OUTPUT_ROOT>/snapshots/<PROJECT_NAME>/<RUN_NAME>_finetune/
```

Its `composed_config.yaml` and `last.ckpt` are the pair intended for inference.
The estimated compression ratio in the training log is based on learned entropy
probabilities; use the inference notebook to measure complete serialized artifact
size.

### Smoke test

Use the following settings to validate paths, model construction, checkpoint
handoff, and both training stages with modest resources:

```python
IN_ZOOMS = [1, 3, 5]
MODEL_COMPLEXITY = 0.125
MAX_STEPS = 4
```

This is a pipeline check, not a useful scientific training budget.

## `fieldspace_inference.ipynb`

### Purpose

This notebook performs inference only. It uses a pretrained model to create real
entropy-coded artifacts, reloads them from disk for standalone decoding, and
measures the compression/fidelity trade-off. It does not train or fine-tune the
model.

The model represents each normalized field as a coarse HEALPix component plus
finer residuals. The analysis transform produces latent `y`; hyper-analysis
produces hyperlatent `z`. `z` is coded first and predicts the conditional
distribution used to code `y`. The artifact also retains passthrough coarse data,
sample configuration, and embedding information needed by the decoder.

### Controls to edit

- `OUTPUT_ROOT`: parent directory for evaluation bundles.
- `SHARD_SIZE`: number of sample records in each `.pt` shard.
- `COMPACT_ARTIFACT_FORMAT`: when `True`, batches side information, stores common
  metadata once, and concatenates entropy streams with offset tables. `False`
  retains the legacy record-per-sample representation.
- `DEVICE_POLICY`: `auto`, `cuda`, or `cpu`.
- `NUM_WORKERS`: use zero for the safest interactive behavior.
- `RANDOM_SEED`: reproducible sampling for diagnostics.
- `MODEL_PROFILES`: registry of inseparable checkpoint/config/data definitions.
- `MODEL_PROFILE`: registry key to run.

A profile defines its label, fine-tuned `composed_config.yaml`, matching
checkpoint, input stores, optional timestep/level selections, and expected
zooms. The group names, variable order, selected depths, and zooms must match the
saved configuration. The training notebook prints a ready-to-paste profile.

### What it does, section by section

1. It sets writable caches, imports the active FieldSpaceNN installation, and
   reports the available PyTorch device.
2. It composes the saved Hydra config, applies only safe data-source selections,
   and validates the entire model profile before allocating the network.
3. It reconstructs the Lightning module, restores checkpoint weights, moves the
   model to the selected device, switches to evaluation mode, and builds
   CompressAI cumulative-distribution tables.
4. It recreates the saved test loader with batch size one, displays encoder input
   shapes, and plots both composed and decomposed normalized inputs.
5. It compresses samples incrementally. Every record stores entropy streams for
   `y` and `z`, shape/config metadata, embedding side information, and configured
   coarse passthrough values. Records are flushed in bounded-size shards.
6. It writes `manifest.json` with profile identity, selected indices, shard
   inventory, raw reference size, timings, device, and format version.
7. It reloads every shard from disk, verifies the manifest/profile selection,
   decodes the hyperprior and primary latent, composes the finest HEALPix output,
   denormalizes each field, and accumulates metrics without keeping every
   reconstruction in memory.
8. It reports complete-artifact compression ratio and bits per HPX value. The
   artifact size includes entropy streams, passthrough arrays, metadata, and
   PyTorch container overhead. Fidelity output contains per-variable RMSE, MAE,
   bias, and Pearson correlation.
9. It reloads one selected sample and plots physical ground truth,
   reconstruction, and signed error on Mollweide maps.
10. The optional deeper diagnostics compare value distributions and sampled
    pixels, calculate angular power spectra, and place the model on an RMSE versus
    compression plot beside three lower-resolution nearest-parent baselines.
11. It writes a presentation-ready result summary and output inventory.
12. It can decode the shards again into an independent, denormalized
    `decoded_fields.zarr`, restoring source dimension order, selected coordinates,
    attributes, grid mapping, and provenance.

### Outputs

For the default root, a profile writes to:

```text
evaluations/hackathon_hyperprior/<MODEL_PROFILE>/
├── manifest.json
├── shard_*.pt
├── metrics_per_variable.csv
├── metrics_summary.json
├── input_zoom_preview.png
├── input_zoom_decomposition.png
├── reconstruction_maps.png
├── deep_diagnostics.png          # after the optional diagnostics cell
├── rmse_vs_compression.png       # after the optional diagnostics cell
└── decoded_fields.zarr/          # after the export cell
```

Known outputs for the selected profile are refreshed on a new compression run;
unrelated files in the directory are preserved. The decoded-Zarr overwrite flag
applies only to the named reconstruction store, which is prepared through a
temporary store before replacement.

### Interpreting the scorecard

The raw-size reference is the number of normalized float32 values at the finest
HEALPix level multiplied by four bytes. The numerator does not include source
Zarr metadata or lower-resolution duplicate inputs. The compressed denominator
is the full size of every serialized shard, so reported compression includes the
cost of everything required to decode.

HEALPix cells have equal area, making the pixelwise means in the dashboard global
area-weighted means. The total error row pools variables even if their physical
units differ; use per-variable rows when judging scientific suitability. The
optional angular spectra complement pointwise metrics by showing whether spatial
variance is preserved across scales.

## Operational checklist

Before training:

- verify the kernel imports all dependencies;
- confirm source variables, level selections, and HEALPix cell count;
- use a unique project/run name unless overwrite is intentional;
- run the low-resolution smoke test before committing high-resolution resources.

Before inference:

- use the fine-tuned checkpoint and configuration from the same run;
- confirm that the normalization JSON referenced by the config still exists;
- check profile variable order, group membership, depths, zooms, and paths;
- start with one timestep and a small shard size;
- inspect reconstruction and error maps before interpreting the aggregate ratio.
