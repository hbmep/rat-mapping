Electrode position, size, and orientation determine efficacy of cervical epidural stimulation to recruit forelimb muscles in rats
=====

This repository has code to reproduce results in the manuscript [Electrode position, size, and orientation determine efficacy of cervical epidural stimulation to recruit forelimb muscles in rats](https://doi.org/10.1101/2025.09.05.674051).

It uses the [hbmep v0.7.0](https://github.com/hbmep/hbmep/releases/tag/v0.7.0). See [pyproject.toml](https://github.com/hbmep/rat-mapping/blob/main/pyproject.toml) for dependencies.

Installation
---------------

Begin by creating a virtual environment.

```bash
    python3.11 -m venv .venv
```

Note that the above command uses Python 3.11. If you have a different version of Python, you can use [conda](https://conda.io) to create a new environment with the required version of Python.

```bash
    conda create -n python-311 python=3.11 -y
    conda activate python-311
    python -m venv .venv
    conda deactivate
```

We can then install in editable mode.

```bash
	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -e .
```

Now, the Python interpreter should be located at ``.venv/bin/python``. You can use this to run the scripts in the [notebooks](https://github.com/hbmep/rat-mapping/tree/main/notebooks) directory.

Citation
-----------

Please cite [Pascual-Leone et al., 2025](https://doi.org/10.1101/2025.09.05.674051) if you find this useful in your research. The BibTeX entry for the paper is::

    @article{pascual-leone_electrode_2025,
        title = {Electrode position, size, and orientation determine efficacy of cervical epidural stimulation to recruit forelimb muscles in rats},
        author = {Pascual-Leone, Andrés and Tyagi, Vishweshwar and Asan, Ahmet S. and Rocha-Flores, Pedro E. and Rodriguez-Lopez, Ovidio and Voit, Walter E. and McIntosh, James R. and Carmel, Jason B.},
        publisher = {bioRxiv},
        year = {2025},
        doi = {10.1101/2025.09.05.674051}
    }

Additionally, you can cite [Tyagi et al., 2024](https://doi.org/10.48550/arXiv.2407.08709) if you find `hbmep` useful in your research. The BibTeX entry for the paper is::

    @article{tyagi_hierarchical_2024,
        title = {Hierarchical {Bayesian} estimation of motor-evoked potential recruitment curves yields accurate and robust estimates},
        author = {Tyagi, Vishweshwar and Murray, Lynda M. and Asan, Ahmet S. and Mandigo, Christopher and Virk, Michael S. and Harel, Noam Y. and Carmel, Jason B. and McIntosh, James R.},
        journal={arXiv preprint arXiv:2407.08709},
        year = {2024},
        doi = {http://doi.org/10.48550/arXiv.2407.08709}
    }
