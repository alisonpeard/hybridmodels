#!/bin/zsh

conda env create -f data-collection-2.yml --experimental-solver=libmamba
conda activate hybridmodels-mfe
# python -m ipykernel install --user --n hybridmodels

python /Users/alison/Documents/DPhil/hybridmodels/python/mfe.py
