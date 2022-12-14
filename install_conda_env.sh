#!/bin/bash

# Installs conda environment for tutorials

conda create --name infinity-tutorials python=3.9
conda activate infinity-tutorials
pip install -r requirements.txt

# TODO: Add arguments for removal
#conda deactivate
#conda remove --name infinity-tutorials --all
