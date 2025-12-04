<p align="center">
  <h1 align="center">OSMa-Bench: Evaluating Open Semantic Mapping Under Varying Lighting Conditions</h1>
  <p align="center">
    <a href="https://github.com/warmhammer/"><strong>Maxim Popov</strong></a>
    ·
    <a href="https://github.com/lumalfo/"><strong>Regina Kurkova</strong></a>
    ·
    <a href="https://github.com/MikhailIum/"><strong>Mikhail Iumanov</strong></a>
    ·
    <a href="https://github.com/JaafarMahmoud1/"><strong>Jaafar Mahmoud</strong></a>
    .
    <a href="https://en.itmo.ru/en/viewperson/464/Sergey_Kolyubin.htm/"><strong>Sergey Kolyubin</strong></a>
  </p>
  <h2 align="center">IROS 2025</h2>
  <h3 align="center"><a href="http://arxiv.org/pdf/2503.10331">Paper</a> | <a href="http://arxiv.org/abs/2503.10331">ArXiv</a> | <a href="https://youtu.be/HqpZ1gDhvKU">Video</a> | <a href="https://huggingface.co/papers/2503.10331">HuggingFace</a> |<a href="https://be2rlab.github.io/OSMa-Bench/">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://be2rlab.github.io/OSMa-Bench/static/images/full_pipeline.png" alt="Pipeline" width="100%">
  </a>
</p>
<p align="center">
</p>
<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#todo">TODO</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#semantic-segmentation-evaluation">Semantic Segmentation Evaluation</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>


## TODO
- [x] Release Habitat Data Generator
- [x] Release semantic segmentation evaluation module
- [ ] Release Visual Question Answering module
- [x] Release Augmented ReplicaCAD dataset
- [x] Release OSMa-Bench dataset

## Data Preparation

We offer [**HaDaGe** (Habitat Data Generator)](https://github.com/warmhammer/habitat_data_generator), a Habitat-based data generator that supports multiple scene datasets. Currently, it is compatible with **Replica**, **ReplicaCAD**, and **HM3D** datasets, and it can be easily adapted to any dataset supported by Habitat.

We have enhanced the **ReplicaCAD** dataset with improved semantic annotations and published it as the [**Augmented ReplicaCAD**](https://huggingface.co/datasets/warmhammer/Augmented_ReplicaCAD_dataset) as part of our benchmark.

Please follow the readme of these packages if you want to prepare the data yourself.

Using **HaDaGe**, we generated the [**OSMa-Bench dataset**](https://huggingface.co/datasets/warmhammer/OSMa-Bench_dataset), which builds upon two base datasets:

* **Augmented ReplicaCAD**: 22 scenes with 4 lighting configurations and a velocity modifier.
* **Habitat Matterport 3D (HM3D)**: 8 scenes with 2 lighting configurations and a velocity modifier.

The pre-generated OSMa-Bench dataset is ready to use with our benchmark for convenience:

```bash
git xet install
git clone https://huggingface.co/datasets/warmhammer/OSMa-Bench_dataset -b compressed
unzip OSMa-Bench_dataset/data.zip
```

## Semantic Segmentation Evaluation

This repository provides **semantic segmentation (semseg) evaluation** tools.
The `semseg/` directory contains the full evaluation pipeline, including scripts, utilities, and configuration files.

The directory also includes benchmarked methods' repositories, benchmarking results, and Docker setup files with running scripts.

For detailed instructions on setup, execution, and workflow, please refer to the [README](semseg/README.md) inside the `semseg/` directory.

## Citation
Using OSMa-Bench in your research? Please cite:

```bibtex
@inproceedings{popov2025osmabench,
    title={OSMa-Bench: Evaluating Open Semantic Mapping Under Varying Lighting Conditions},
    author={Popov, Maxim and Kurkova, Regina and Iumanov, Mikhail and Mahmoud, Jaafar and Kolyubin, Sergey},
    booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2025}
}
```
