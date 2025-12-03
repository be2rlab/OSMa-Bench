# Open Semantic Mapping Evaluation
This repository stores Docker containers and associated scripts for running open-vocabulary semantic mapping methods. It also includes tools for evaluation and metric calculation. This repo is part of [OSMa-Bench](https://be2rlab.github.io/OSMa-Bench/) pipeline.

## Supported Pipelines:
- [BeyondBareQueries](https://linukc.github.io/BeyondBareQueries/)
- [ConceptGraphs](https://concept-graphs.github.io/)
- [OpenScene](https://pengsongyou.github.io/openscene)

## Data Preparation
The folder hierarchy should be organized as follows:
```bash
data/
    datasets/
        generated/
            replica_cad/
            hm3d/
            ...
        replica/
    gt/
semseg/
```

## Build docker
```bash
make build-<approach_name>
```

## Run
```bash
make run-<approach_name>
```

## Specific settings 
Inside the docker you can edit the export files in export folder, and then run one of them
```bash
bash /export/osma-bench/run_<approach_name>_<dataset>.sh   # approach running
bash /export/osma-bench/gt_reconstruction_<dataset>.sh     # running gt pointcloud reconstruction for eval
bash /export/osma-bench/eval_<approach_name>_<dataset>.sh  # evaluation and metrics calculation
```

## Visualize
```bash
make prepare-terminal-for-visualization
```

## Citing

Using this repo in your research? Please cite the following paper: [arxiv](https://arxiv.org/abs/2503.10331)

```bibtex
@inproceedings{popov2025osmabench,
    title={OSMa-Bench: Evaluating Open Semantic Mapping Under Varying Lighting Conditions},
    author={Popov, Maxim and Kurkova, Regina and Iumanov, Mikhail and Mahmoud, Jaafar and Kolyubin, Sergey},
    booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year={2025}
}
```