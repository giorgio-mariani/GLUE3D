# README

This folder contains Docker images of the various 3D and 2D-LLMs that were evaluated.

To build the Docker images, run:
```bash
bash scripts/build_images.sh
```
This will result in several images, each tagged with the models that they support. See [here](../glue3d/models/loaders.py) for a list of available model names.

> [!NOTE]
> Correct build of all the images contained in this folder was tested on  October 10th, 2025. The building setup was:
> - Operating system: `Ubuntu 24.04.2 LTS`
> - CPU: `AMD Ryzen 5 7600 6-Core Processor`
> - GPU: `GeForce RTX 3090 Ti`
> - RAM: `32 GiB DDR5`



Each image can be run on a specific task using the command:
```bash
bash scripts/run_generation.sh [MODEL] [DATSET] [TASK]

# e.g.
# bash scripts/run_generation.sh pointllm_7B GLUE3D-points-8K binary_task
```
