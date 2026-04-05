# High-Throughput Screening (HTS) Pipeline for Deep Eutectic Solvents (DES)

This repository contains the Python source code and dataset for the systematic benchmarking of classical empirical thermodynamic models against Deep Eutectic Solvents (DES). 

This computational pipeline was developed to evaluate the limits of the Hoftyzer-Van Krevelen (HVK) group contribution method and Eyring's Hole Theory across nearly 1,000 DES combinations.

## Repository Contents
* `DES_Physicochemical_Screening.py` : The main Python script utilizing RDKit and SMARTS-based functional group recognition to compute macroscopic density and dynamic viscosity.
* `Screening_DES_HoleTheory_Physicochemical.csv` : The dataset containing the SMILES strings of Hydrogen Bond Acceptors (HBAs) and Donors (HBDs), along with the computational output.

## Dependencies
To run this script, the following Python libraries are required:
* `rdkit`
* `pandas`
* `numpy`
* `matplotlib`

## Usage
The script is designed to function as an ultra-fast "first-tier negative filter" to identify and eliminate thermodynamically unviable DES candidates prior to applying higher-level quantum mechanical methods (e.g., COSMO-RS).

## Citation
If you use this code or dataset in your research, please cite our paper:
> *(Citation details will be updated upon publication)*
