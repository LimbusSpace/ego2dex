# ego2dex

Egocentric video to dexterous-hand annotations, retargeting, and training-data exports.

ego2dex is a modular research pipeline for turning first-person video into structured data for robot learning. It covers video ingestion, hand pose, object detection and segmentation, hand-object annotations, camera metadata, robot-hand retargeting, export, and visualization.

> Scope: ego2dex produces perception and dataset artifacts. It is not a policy, controller, or robot driver.

## Pipeline

```text
ingest -> camera / pose -> hands -> arms -> detection -> segmentation / tracking
       -> hand-object interaction -> annotations -> retargeting -> export
```

The package provides a common annotation schema and optional stage implementations. Model backends are loaded lazily, so the core package and dry-run pipeline work without a GPU or model weights.

## Repository status

This repository is an independent distribution maintained by **LimbusSpace** and is based on the upstream [adityagarg7/ego2dex](https://github.com/adityagarg7/ego2dex) project. See the commit history and [LICENSE](LICENSE) for copyright and licensing details.

The project is research-oriented and currently in alpha. Some stage adapters are scaffolding or require third-party source code, system packages, model weights, or additional integration work. The supported status of each stage is documented in [docs/architecture.md](docs/architecture.md) and [docs/licenses.md](docs/licenses.md).

## Quick start

The core install and deterministic CPU smoke test do not download model weights:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

ego2dex run --config configs/pipeline/smoke.yaml \
  --input assets/synthetic \
  --output outputs/smoke
```

You can also use Pixi:

```bash
pixi install
pixi run smoke
pixi run test
```

Run the tests that do not require model assets:

```bash
pytest -m "not requires_models"
```

## Optional model environments

Install only the extras required by the stages you intend to use:

```bash
pip install -e ".[mediapipe]"
pip install -e ".[detection,sam2]"
pip install -e ".[hamer,retarget,export]"
```

The full installation notes, backend matrix, system requirements, and known limitations are in [docs/install.md](docs/install.md).

## Weights and local infrastructure

**Model weights, MANO assets, datasets, videos, extracted frames, and generated outputs are never stored in this repository.**

The WSL layout used by this deployment is:

```text
/home/ninewell/src/ego2dex/       # source repository
/home/ninewell/models/ego2dex/     # external model assets
```

The model installer defaults to `~/models/ego2dex`. Override it explicitly when needed:

```bash
export EGO2DEX_DATA_DIR=/home/ninewell/models/ego2dex
bash scripts/install_models.sh mediapipe sam2 grounding_dino
```

See [infra/README.md](infra/README.md). Ignoring a file in Git is not permission to distribute it: contributors must keep weights outside the repository and comply with each model's terms.

## Documentation

- [Installation and environments](docs/install.md)
- [Architecture and stage lifecycle](docs/architecture.md)
- [Annotation schema](docs/annotation_schema.md)
- [Dataset compatibility](docs/datasets.md)
- [Retargeting](docs/retargeting.md)
- [Model survey and licenses](docs/sota_survey.md)
- [Third-party license matrix](docs/licenses.md)
- [Pretraining/export pathways](docs/pretraining.md)

## Development

```bash
make check
make test
```

The CI workflow runs formatting, linting, type checks, and CPU tests without downloading model weights.

## License

The original ego2dex source code is released under the MIT License. This license applies only to code covered by this repository's copyright notices. Third-party libraries, model code, model weights, datasets, MANO assets, robot descriptions, and generated data have their own licenses and terms. See [LICENSE](LICENSE) and [docs/licenses.md](docs/licenses.md) before redistribution or commercial use.

