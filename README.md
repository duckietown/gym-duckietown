## Installation
```bash
git clone https://github.com/mphamhung/GAIL
cd GAIL
git submodule update --recursive --init
conda env create -f environment.yaml
conda activate gail-env
```
## add conda env to jupyter
```bash
python -m ipykernel install --user --name=gail-env
```
## start jupyter 
```bash
jupyter notebook
```
