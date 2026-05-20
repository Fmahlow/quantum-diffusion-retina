# Notebook workflow

The original notebook is preserved at the project root. New notebooks should call the code in `src/retina_synthesis` and keep only narrative, plots, and experiment-specific parameters.

Recommended split:

1. `00_data_audit.ipynb`: dataset inspection, labels, class balance, and visual sanity checks.
2. `01_classical_ddpm_baseline.ipynb`: class-specific DDPM training commands and generated samples.
3. `02_quantum_generator.ipynb`: quantum or hybrid quantum-classical generator experiments.
4. `03_evaluation.ipynb`: metric tables, confidence intervals, downstream classifier tests, and figures for the paper.

