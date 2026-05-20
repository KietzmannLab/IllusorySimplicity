# Setup Instructions

## Python Environment

Dependencies are listed in the environment.yml file in this repository and can be installed with conda.

Local dependencies (`ephyslib` and `macaquethings`) are included in this repository under `./packages`.
These can be installed into the conda environment using:

`pip install -e packages/ephyslib`

`pip install -e packages/macaquethings`

## Repository Structure

### Data

All datasets should live in `./datasets`

### Results

Results are saved to `./results`

### Scripts

All analysis scripts are located in `./scripts`

## Submission

Submission scripts for analyses with parameters for different conditions are located in `./submit`
