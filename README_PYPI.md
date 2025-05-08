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
- [**OpenAI API key**](https://platform.openai.com/api-keys)
- [**LangSmith API key**](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key)

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

3. Configure your API keys

   **MI-Agent** will automatically look for a file named `.env` in your current working directory (or any parent) and load any keys it finds.  

   In the folder where you’ll run the CLI (or in any ancestor), create a file called **`.env`** containing:

   ```bash
   OPENAI_API_KEY=sk-…
   LANGCHAIN_API_KEY=lsv2_…
   ```

4. Prepare your problem file
   
   MI-Agent requires a `.txt` file (an example is provided in the `sample_problem.txt` in the project root of the [source code](https://github.com/hasan-sayeed/mi_agent)) which contains:

   - your problem description

   - relative paths to your CSV(s), **including any folder prefix** (e.g. `data/sample_data.csv`)

   Example `problem.txt`:
   ```bash
   You are tasked with predicting alloy strength from composition data...

   - data/sample_data_1.csv: Contains experimental results...
   - data/sample_data_2.csv: Contains formulation recipes...
   ```

5. Run the agent

   Now, start the `mi_agent` pipeline as below:
   ```bash
   mi_agent --problem-file <path/to/problem.txt> --output-dir <path/to/output_dir>
   ```

   **MI-Agent** will:

   - Identify & load the CSV(s) listed in the problem file
   - Merge files if needed
   - Select target & features
   - Propose & execute EDA
   - Save all generated code (`*.py`) for EDA analysis and images (`*.png`) generated during EDA into <output_dir>
   - Run multiple ML models, select top 5, tune hyperparameters, and choose the best model
   - Generate and save a 5-page technical summary into <output_dir>
   - Log all reasoning steps to LangSmith