This repository contains the code, figure, and derived data used to reproduce the analyses, figures, and tables for the accompanying paper. Analyses are run from a single Jupyter notebook that calls modular Python functions in Functions/ to load derived datasets (.pkl), perform preprocessing, compute summary metrics, and generate publication-ready plots and tables. The repository is structured to allow end-to-end reproduction from environment setup through Figure/table regeneration.

Code is licensed under MIT. 
Data and figures are licensed under CC-BY 4.0 unless otherwise noted.

System requirements
- OS: macOS (developed on macOS; should also run on Linux; Windows untested)
- Python: 3.10+ (recommended: 3.11)
- Disk: ~5 GB (depends on whether you download optional raw data)
- RAM: recommended ≥ 16 GB (will run with less depending on dataset size)
- Optional: a modern CPU; no GPU required
