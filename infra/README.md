# ego2dex infrastructure

Model weights, datasets, raw videos, extracted frames, and generated outputs must live outside this Git repository and are never contributed as source files.

## Canonical WSL layout

/home/ninewell/src/ego2dex/       # repository: code, configs, tests, docs
/home/ninewell/models/ego2dex/     # external model weights and downloaded assets

The installer uses ~/models/ego2dex by default. Override it with EGO2DEX_DATA_DIR for another machine or storage volume.

Contributions contain reproducible code, lightweight configuration, tests, and documentation. Machine-specific downloads belong to contributor infrastructure, never Git.
