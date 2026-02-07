# How to run this project:

## Install Git LFS for embeddings
### On Linux
```bash
sudo apt update
sudo apt install git-lfs
git lfs install
git lfs pull
```

## Create venv:
```bash
python3 -m venv .venv
```

## Activate venv:
### On Linux
```bash
source .venv/bin/activate
```
### On Windows
```bash
.\.venv\Scripts\Activate
```

## Install ipykernel
```bash
pip install ipykernel
```

## Register kernel
```bash
python -m ipykernel install --user --name ml-project
```

## Install python libraries
```bash
pip install -r requirements.txt
```

<!-- ## Install ??? TODO
```bash
python -m spacy download en_core_web_sm
``` -->