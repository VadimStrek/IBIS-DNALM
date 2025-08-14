# Using DNA language models for the IBIS Challenge
This repository contains scripts for obtaining a solution for the IBIS Challenge (2024) using DNA language models DNABERT-2, GENA-LM and Nucleotide Transformer (NT).

Scripts for GENA-LM and Nucleotide Transformer are run through Jupyter Notebook, for DNABERT-2 are run through command-line.
## Setup environment
### For GENA-LM and NT
`conda create -n env python=3.12.2
conda activate env
python3 -m pip install -r requirements.txt`
### For DNABERT-2
`conda create -n dna python=3.8.20
conda activate dna
cd DNABERT2
python3 -m pip install -r requirements.txt`
## Data
Data packages used in the challenge are available at the link: https://zenodo.org/records/15056803

For datasets assembling you should use the bibis package (https://github.com/autosome-ru/ibis-challenge)
## Fine-tuning
### For GENA-LM and NT
a
### For DNABERT-2
a
