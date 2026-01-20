# 3D-Scene QA Pipeline

Main contributor: <a href="https://github.com/lumalfo/"><strong>Regina Kurkova</strong></a>

> End-to-end pipeline that  
> 1. selects frames from a simulation 
> 2. generates dense scene descriptions
> 3. auto-creates QA pairs
> 4. validates / de-duplicates them
> 5. answers questions with a scene-graph
> 6. evaluates the whole run with accuracy metrics.

---

## Repository layout

```
src/
├── config.py                     # simple YAML → object loader
├── generation/                   # ⇣ Stage-1 : data generation
│   ├── text_desc_generation.py
│   └── qa_generation.py
├── validation/                   # ⇣ Stage-2 : QA validation
│   ├── qa_validation.py
│   └── validation_utils.py   
└── evaluation/
    └── scene_graph_answering.py  # ⇣ Stage-3 : answer via graph
    └── graph_evaluation.py       # ⇣ Stage-4 : compute metrics
utils/
├── api.py            # post_with_retry, request_gemini, …
├── json_utils.py     # load_json, save_json, clean_json_response, …
└── parsing.py        # tiny helpers (infer_answer_type, etc.)
run_pipeline.sh       # one-click launcher
config/gemini_qa.yml  # central YAML with prompts & API keys
```

---

## Configuration (`config/gemini_qa.yml`)

| key | meaning |
|-----|---------|
| `vlm_prompt`, `llm_prompt` | high-level system prompts |
| `gemini_api_key`, `url`, `vlm`, `llm` | Gemini API settings |
| `qa_generation_prompt`, `validation_prompt` | prompts used internally |
| `base_scenes_dir` | root for every scene (`./data` by default) |
| `rejection_keyword` | special token that marks a frame as “blocked” |
| `frame_step`, `selection_threshold` | heuristics for frame sampling |

---

## Stage 1 · Generation

| script | role |
|--------|------|
| `text_desc_generation.py` | parses `traj.txt`, selects frames, queries VLM, generates `<scene>_descriptions.json` |
| `qa_generation.py` | builds object inventory, queries LLM for QAs, writes `<scene>_questions.json` |

Manual mode:
```bash
python -m src.generation.text_desc_generation config/gemini_qa.yml \
       --scene my_scene --manual
```

---

## Stage 2 · Validation

→ Produces: `<scene>_validated_questions.json`

Logs:
```
vqa/validation_process.log
vqa/removed_questions.log
vqa/filtered_objects_{before,after}.txt
```

---

## Stage 3 · Scene-graph answering

```bash
python -m src.scene_graph_answering \
       -c config/gemini_qa.yml \
       --questions data/<scene>/vqa/<scene>_validated_questions.json \
       --graph     graphs/<scene>/scene_graph.json \
       --output    output/<scene>_answered.json
```

This sends the scene graph and QA in batches to Gemini and appends a `"scene_graph_answer"`.

---

## Stage 4 · Evaluation

`evaluation/graph_evaluation.py`:

- Exact match for yes/no and numbers  
- LLM-based judging for other answers  
- Produces `evaluated/<scene>.json` + metrics in `evaluated/metrics.csv`