This repo was developed by Ouns El Harzli (@ouns972) and Hugo Wallner (@hugo1717) (equal contribution).


# Decoder-only Supervised Sparse Auto-Encoders (SSAEs) 🔧

A research repo for learning decoder-only SSAEs that reconstruct text embeddings (Stable Diffusion 3.5 text encodings based on T5) from concept dictionry. The library contains components to generate prompts, extract embeddings (from SD3), build HDF5-based datasets, train decoder and sparse latent features, and perform inference to reconstruct embeddings usable for conditional image generation.

---

## ✅ Quick summary

- Core functionality: dataset generation -> embedding extraction -> training decoder + sparse features -> produce new embeddings -> inference for image synthesis.
- Main entry points:
  - `dataset_generation/functions.py` — generate combinatorial prompts from category/property lists
  - `get_embeddings_large_turbo_many_h5.py` — extracts text embeddings using SD3 pipeline and saves HDF5 files
  - `training_cli.py` / `trainable_inputs_all_clips.py` — training loop
  - `inference/*.py` — inference helpers (load model, reconstruct embeddings)

---

## 🔧 Requirements & setup

1. Clone the repo:

```bash
git clone <repo-url>
cd decoder-only-ssae
```

2. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: The repo uses PyTorch, Diffusers and SD3 models — a GPU is strongly recommended for embedding extraction and training.

---

## 🧱 Data layout (expected)

The dataloader expects a `folder_path` with the following structure:

```
<folder_path>/
  prompts.json          # list of {"id":.., "prompt":..}
  properties.json       # categories -> property lists
  embds/                # generated embeddings
    embds_0/            # for prompt 0
      embds.h5
      embds_pooled.h5   # (optional)
      prompts.txt
    embds_1/
    ...
```

`trainings/config/params_default.yaml` contains default settings (dataloader path, training hyperparams, etc.).

---

## ▶️ Typical workflow

1. Generate prompts (combinatorial from categories):

```bash 
python generate_prompts.py
```



2. Extract embeddings (uses Stable Diffusion 3 transformer / pipeline):

```bash
python get_embeddings_large_turbo_many_h5.py
# edit the top of the script to point to your prompts folder, model id and device
```

3. Train a model:

```bash
python training_cli.py --output_folder results/training1 --path_yaml trainings/config/params_default.yaml --overwrite_output True
```

This will save `model.pt` and `all_params.yaml` in the output folder and write logs via the repository's `Logger` implementation.

4. Inference — use the notebook 🔍

For interactive inference and image generation we recommend using the notebook `inference/notebooks/inference_and_testing_output_visuals.ipynb`. The notebook demonstrates step-by-step how to:

- Load the inference helper and the image generator:

```python
from inference.inference_model_avg import SFDInferenceModelAvg
from inference.image_generation.image_generator import ImageGenerator
sdf_inference = SFDInferenceModelAvg('results/training_cigarettes_ouns')
image_generator = ImageGenerator(simulated=False, device=device)
```

- Search for a prompt (by properties) and generate the original image from the prompt:

```python
idx1, prompt1 = sdf_inference.search_idx_prompt([...])
image_generator.generate_image_from_prompt(prompt=prompt1, image_name='image_blond_bar_holding_gun.png')
```

- Reconstruct the embedding and generate a new image from the reconstructed embedding:

```python
embd1 = sdf_inference.get_x(idx=idx1)
embd_1, pooled_embd_1 = sdf_inference.overwrite_full_embedding(embd1, idx=idx1)
image_generator.generate_image_from_embd(
    prompt_embeds=embd_1.detach(),
    pooled_prompt_embeds=pooled_embd_1.detach(),
    image_name='image_blond_bar_holding_gun_reconstructed.png',
)
```

The notebook also shows how to make manual edits to the mask and to `decoder.Y` to craft custom embeddings before calling the generator (see the notebook cells for the exact examples).



