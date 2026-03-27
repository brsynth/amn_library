# AMN
Artificial Metabolic Networks
| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-amn-green.svg)](https://anaconda.org/conda-forge/amn) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/amn.svg)](https://anaconda.org/conda-forge/amn) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/amn.svg)](https://anaconda.org/conda-forge/amn) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/amn.svg)](https://anaconda.org/conda-forge/amn) |

# Supporting content for the Reservoir Computing paper with Bacteria paper

[![Github Version](https://img.shields.io/github/v/release/brsynth/molecule-signature-paper?display_name=tag&sort=semver&logo=github)](version)
[![Github Licence](https://img.shields.io/github/license/brsynth/molecule-signature-paper?logo=github)](LICENSE.md)

This repository contains code to support the Computing paper with Bacteria publication. See citation for details.

## Table of Contents
- [1. Repository structure](#1-repository-structure)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
- [4. Citation](#4-citation)

## 1. Repository structure

```text
.
├── Dataset_input       < placeholder for data files >
│   └── ..
├── Reservoir       < trained reservoir model>
│   └── ..
├── Result     
│   └── ..
├── Library       < supporting code for notebook >
│   └── ..
├── 1.Dataset-species.ipynb
├── 2.Fixed-prior.ipynb
├── 3.ML-covid.ipynb
├── 4.Reservoir-covid.ipynb
├── 5.Reservoir-species.ipynb
├── README.md
└── requirements.yaml


```
## 2. Installation

The following steps will set up a `reservoir` conda environment.

0. **Install Conda:**

    The conda package manager is required. If you do not have it installed, you
    can download it from [here](https://docs.conda.io/en/latest/miniconda.html).
    Follow the instructions on the page to install Conda. For example, on
    Windows, you would download the installer and run it. On macOS and Linux,
    you might use a command like:

    ```bash
    bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
    ```

    Follow the prompts on the installer to complete the installation.

1. **Install dependencies:**

    ```bash
    conda env create -f requirements.yaml
    conda activate reservoir
    ```

2. **Download data:**

    Trained reservoir models and most important datasets are available as a Zenodo archive: <https://doi.org/10.5281/zenodo.14961168>. Extract the files and place them in the `Dataset-input`, `Reservoir`, `Result` directory.
## 3. Usage

 **Dataset-sp:**


## 4. Citation
   