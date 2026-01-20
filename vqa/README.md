# Visual Question Answering

Main contributer: <a href="https://github.com/lumalfo/"><strong>Regina Kurkova</strong></a>\
Refactored by: <a href="https://warmhammer.github.io/"><strong>Maxim Popov</strong></a>

<p align="center">
  <a href="">
    <img src="https://be2rlab.github.io/OSMa-Bench/static/images/vqa_pipeline.png" alt="VQA Pipeline" width="100%">
  </a>
</p>
<p align="center">
</p>


This repository contains a pipeline for generating, validating, and answering scene-based questions using scene graphs. The process is organized into three main steps: **generation**, **scene graph answering**, and **evaluation**. This dataset is part of [OSMa-Bench](https://be2rlab.github.io/OSMa-Bench/) pipeline.

## Installation

You can run VQA module generation using either **Gemini API** or the **Ollama**.
To use Ollama, follow the official [installation guide](https://ollama.com/download).

Install the required dependencies:

```bash
pip install -r requirements.txt   # Recommended to run inside a conda environment
```

## Pipeline Overview

For each scene in the dataset, the pipeline executes these steps:

1. **Generation:** Descriptions → QA → Validation
2. **Scene Graph Answering**
3. **Evaluation**

## Running the Pipeline

To run the full pipeline, simply execute:

```bash
bash run_pipeline.sh
```

This script processes all scenes sequentially.
If you want to run individual steps, please follow the structure of the run_pipeline.sh file.

## Citing VQA

Using VQA in your research? Please cite following paper: [arxiv](https://arxiv.org/abs/2503.10331).

```bibtex
@inproceedings{popov2025osmabench,
    title     = {OSMa-Bench: Evaluating Open Semantic Mapping Under Varying Lighting Conditions},
    author    = {Popov, Maxim and Kurkova, Regina and Iumanov, Mikhail and Mahmoud, Jaafar and Kolyubin, Sergey},
    booktitle = {2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year      = {2025}
}
```