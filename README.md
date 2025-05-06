[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/MI-Agent.svg?branch=main)](https://cirrus-ci.com/github/<USER>/MI-Agent)
[![ReadTheDocs](https://readthedocs.org/projects/MI-Agent/badge/?version=latest)](https://MI-Agent.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/MI-Agent/main.svg)](https://coveralls.io/r/<USER>/MI-Agent)
[![PyPI-Server](https://img.shields.io/pypi/v/MI-Agent.svg)](https://pypi.org/project/MI-Agent/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/MI-Agent.svg)](https://anaconda.org/conda-forge/MI-Agent)
[![Monthly Downloads](https://pepy.tech/badge/MI-Agent/month)](https://pepy.tech/project/MI-Agent)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/MI-Agent)
-->

# MI-Agent

An **agentic workflow** for materials-informatics (MI) engineers, built with **LangGraph** and powered by OpenAI models. MI-Agent codifies the end-to-end MI pipeline—data loading, merging, feature selection, EDA, AutoML baselining, hyperparameter tuning, and executive reporting—into reusable nodes orchestrated as a LangGraph. LangSmith integration tracks and visualizes your graph executions. The result? MI workflows that run in seconds instead of hours, boosting your productivity by an order of magnitude.

---

## 🚀 Why MI-Agent?

- **Agentic LangGraph design** lets you hit “play” on a full MI pipeline  
- **10× faster**: eliminate boilerplate and manual scripting  
- **Extensible nodes**: swap in your own extractors, metrics, or plots  
- **LangSmith-backed** for graph tracking, versioning, and observability  
- Production-ready: versionable, testable, pip-installable

---

## 🛠️ Prerequisites

- **Conda** (Miniconda or Anaconda)  
- **Python 3.10**  
- **OpenAI API key**  
- **LangSmith API key**  

---

## Quickstart: Installation via pip

1. Create & activate a conda environment  
   ```bash
   conda create -n mi_agent python=3.10 -y
   conda activate mi_agent
   ```

2. Install via pip
   ```bash
   pip install mi_agent
   ```

3. Configure your API keys **for this session**

   _You’ll need to re-enter these each time you open a new terminal._
   
   MI-Agent reads **only** from real environment variables. Set them in your shell before running:
   
   Windows PowerShell:
   ```bash
   $Env:OPENAI_API_KEY = "sk-…"
   $Env:LANGCHAIN_API_KEY = "lsv2_..."      <---- your LangSmith API key
   ```

   macOS/Linux (bash, zsh):
   ```bash
   export OPENAI_API_KEY ="sk-…"
   export LANGCHAIN_API_KEY="lsv2_..."      <---- your LangSmith API key
   ```

4. Prepare your problem file
   
   MI-Agent requires a `.txt` file (an example is provided in the `sample_problem.txt` in the project root) which contains:

   - your problem description

   - relative paths to your CSV(s), **including any folder prefix** (e.g. `data/sample_data.csv`)

   Example `problem.txt`:
   ```bash
   You are tasked with predicting alloy strength from composition data...

   - data/sample_data_1.csv: Contains experimental results...
   - data/sample_data_2.csv: Contains formulation recipes...
   ```

5. Run the agent

   Now, invoke `mi_agent …` in the same terminal session you entered your API keys:
   ```bash
   mi_agent --problem-file <path/to/problem.txt> --output-dir <path/to/output_dir>
   ```

   MI-Agent will:

   - Identify & load the CSV(s) listed in the problem file
   - Merge files if needed
   - Select target & features
   - Propose & execute EDA
   - Save all generated code (`*.py`) for EDA analysis and images (`*.png`) generated during EDA into <output_dir>
   - Run AutoML baseline + hyperparameter tuning
   - Emit a two-page executive summary
   - Log every step to LangSmith


## From source: clone & run

1. Clone the repo
   ```bash
   git clone https://github.com/hasan-sayeed/mi_agent.git
   cd mi_agent
   ```

2. Create & activate the conda env from `environment.yml`
   ```bash
   conda env create -f environment.yml -n mi_agent
   conda activate mi_agent
   ```

3. Install in editable mode
   ```bash
   pip install -e .
   ```

4. Configure your API keys **for this session**

   _You’ll need to re-enter these each time you open a new terminal._
   
   MI-Agent reads **only** from real environment variables. Set them in your shell before running:
   
   Windows PowerShell:
   ```bash
   $Env:OPENAI_API_KEY = "sk-…"
   $Env:LANGCHAIN_API_KEY = "lsv2_..."      <---- your LangSmith API key
   ```

   macOS/Linux (bash, zsh):
   ```bash
   export OPENAI_API_KEY ="sk-…"
   export LANGCHAIN_API_KEY="lsv2_..."      <---- your LangSmith API key
   ```

5. Prepare your problem file as above and then invoke `mi_agent …` in the same terminal session you entered your API keys::
   ```bash
   mi_agent --problem-file <path/to/problem.txt> --output-dir <path/to/output_dir>
   ```

## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── mi_agent            <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Contributing

1. Fork & clone this repository

2. Create a feature branch

3. Implement your node, extractor, or fix & add tests

4. Open a Pull Request, describing your changes and any new configs

We welcome improvements, new extractors, and integrations!

## Feedback & Feature Requests

Your feedback drives MI-Agent’s roadmap!

- Open an issue on GitHub

- Email the maintainers at `hasan.sayeed.utah.edu`

Let’s make materials-informatics ten times faster—together!

## Note

This project has been set up using [PyScaffold] 4.6 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
