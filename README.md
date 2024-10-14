# Murphy et al. (2024)  Quantifying bead accumulation inside a cannibalistic macrophage population

Preprint: TBC

This repository holds the key Julia code used to generate figures in the manuscript.

Please contact Ryan Murphy for any queries or questions.

Code developed and run in October 2024 using:

- Julia Version  1.9.3 (see https://julialang.org/downloads/ )
- Julia packages: Plots, LinearAlgebra, NLopt, .Threads, Interpolations, Distributions, Roots, LaTeXStrings, DifferentialEquations, CSV, DataFrames, Random, StatsPlots.

## Guide to using the code
The script InstallPackages.jl can be used to install packages (by uncommenting the relevant lines). Scripts are summarised in the table below.


| | Script        | Figures in manuscript | Short description           | 
| :---:   | :---: | :---: | :---: |
|1| Figs_SyntheticData.jl  | Figures 2-4 | Code to perform synthetic data study |
|2| Figs_ExperimentalData.jl     | Figures 5-7  |  Code to analyse experimental data   |  
