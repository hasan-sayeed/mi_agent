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

## Installation via pip

1. Create & activate a conda environment  
   ```bash
   conda create -n mi_agent python=3.10 -y
   conda activate mi_agent
   ```

2. Install via pip
   ```bash
   pip install materials_informatics_agent
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