# transformer-research-code

Transformer training for DE→EN machine translation.

## Setup

**Recommended (conda, one command):**

```bash
bash scripts/setup_env.sh          # creates env named "transformer"
export PYTHONNOUSERSITE=1          # important on shared HPC systems
conda activate transformer
```

**Pip-only (if you already have a venv/conda env):**

```bash
# GPU — use cu121 on PSC Bridges2 V100 nodes; plain "pip install torch" pulls CUDA 13 and will fail.
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Verify GPU access before a long run:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Train

```bash
python train_model.py --dataset_name toy --batch_size 128 --epochs 3
```

Config defaults live in `configs/default.yaml`. Override via CLI flags or by editing that file.

## Dev tools (optional)

```bash
pip install -r requirements-dev.txt
```
