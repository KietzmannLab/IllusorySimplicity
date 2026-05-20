# Setup Instructions

## Python Environment

Dependencies are listed in the environment.yml file in this repository and can be installed with conda.

Local dependencies (`ephyslib` and `macaquethings`) are included in this repository under `./packages`.
These can be installed into the conda environment using:

`pip install -e packages/ephyslib`

`pip install -e packages/macaquethings`

## Repository Structure

All paths set in scripts are relative to the root of this repository and therefore should be run from the root of this repository.
(e.g. `python scripts/allMUA_dprime_new.py`. Scripts may not work if run from the scripts directoryt directly).

Different runtimes for jupyter notebooks may have different conventions for setting the working directory for the Jupyter Kernel.
Included notebooks set paths relative to the location of each `.ipynb` file. This is appropriate for manually launched jupyer lab servers. 
Editors like VSCode may launch the kernel in the repository root directory instead. In this case paths at the top of each notebook must be adjusted to be relative to the repository root.

### Data

All scripts expect datasets to be located in `./datasets`.

### Results

Results are saved to `./results`

### Scripts

All analysis scripts are located in `./scripts`

## Submission

Submission scripts for analyses with parameters for different conditions are located in `./submit`
