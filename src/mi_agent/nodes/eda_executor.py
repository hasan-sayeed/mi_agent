"""Node: generate & run EDA code, then explain results."""
import os
import base64
import threading
from IPython.display import display, Image
from mi_agent.extractors import eda_code_extractor, eda_explainer
from mi_agent.utils import ensure_dir, move_file, is_code_error
from mi_agent.app_config import settings
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import HumanMessage
from mi_agent.states import EDAExecutionState

_matplotlib_lock = threading.Lock()

class EDAExecutor:
    """Wrap run_eda_code & explain_eda_result logic."""

    @staticmethod
    def run_eda_code(state: EDAExecutionState) -> dict:
        """Invoke LLM to write a one-off EDA script, run it, save code+image under data/."""
        cfg = state["plan"].execution_state
        eda_task = f"{cfg['eda_plan_name']}: {cfg['eda_plan_description']}"
        prompt = f"""
                You're a data scientist. Write Python code to perform the EDA on the problem: {cfg['problem_statement']}

                Write a self‐contained Python script that:

                1. Forces Matplotlib into headless mode by doing:
                ```python
                import matplotlib
                matplotlib.use("Agg")```
                before importing `matplotlib.pyplot`.
                2. Imports any other needed libraries (pandas, matplotlib.pyplot as plt, seaborn, etc.). 
                3. Loads the data from {cfg['file_paths']} into a DataFrame. 
                4. Perform exactly one clear and simple EDA task: {eda_task} on the target {cfg['target_column']} and features {cfg['feature_columns']}. 
                5. Plot exactly one plot (where necessary). Ensure the figure dimensions are clear, e.g.:
                ```plt.figure(figsize=(8,6))
                ```
                6. After plotting:
                ```plt.savefig("{cfg['eda_plan_name']}.png")
                plt.close()
                ```
                7. Do NOT use `plt.show()`.
                Include any printing of summary statistics or DataFrame heads you think are appropriate, but ensure the script ends by writing one PNG file.
                """
        code = eda_code_extractor.invoke(prompt)["responses"][0].code
        # ensure folder
        ensure_dir(settings.output_dir)
        fname = f"{cfg['eda_plan_name'].replace(' ', '_').lower()}.py"
        code_path = os.path.join(settings.output_dir, fname)
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        # run and capture image
        output = None
        success = False
        try_count = 0
        for _ in range(state.get("max_retries", 2)):
            try_count += 1
            try:
                # only one thread at a time may exec matplotlib
                with _matplotlib_lock:
                    repl = PythonREPL()
                    output = repl.run(code)
                success = True
                break
            except Exception as e:
                output = str(e)
        img_name = f"{cfg['eda_plan_name']}.png"
        img_path = move_file(img_name, settings.output_dir) if success and os.path.exists(img_name) else []
        return {
            "generated_code": code,
            "code_output": output,
            "code_file_path": code_path,
            "image_path": img_path,
            "tries": try_count,
        }

    @staticmethod
    def explain_eda_result(state: EDAExecutionState) -> dict:
        """Read the code+output+image and ask LLM to explain."""
        image_info = "No image was generated."
        image_section = ""
        image_base64 = None
        cfg = state["plan"].execution_state

        if state.get("image_path"):
            image_path = state["image_path"][0]
            try:
                with open(image_path, "rb") as f:
                    encoded_image = base64.b64encode(f.read()).decode("utf-8")
                    image_base64 = encoded_image
                    image_info = "An image was generated and included below."
                    image_section = f"![EDA Plot](data:image/png;base64,{encoded_image})"
            except Exception as e:
                image_info = f"An image was expected at {image_path}, but could not be read: {e}"

        output = state.get("code_output", "").strip()
        error_occurred = is_code_error(output)

        prompt = f"""
                    You are an experienced data scientist. Your job is to interpret and explain EDA results in clear, simple language for a user who is not a data science expert.

                    Problem Statement:
                    {cfg['problem_statement']}

                    EDA Task:
                    {cfg['eda_plan_name']}: {cfg['eda_plan_description']}

                    Generated Code:
                    {state['generated_code']}

                    Code Output:
                    {output}

                    Image Info:
                    {image_info}
                    {image_section}
                    """

        if error_occurred:
            prompt += "\nAn error occurred during execution. Explain what went wrong and how the user might fix it — again, in friendly, easy language."
        else:
            prompt += "\nThe code executed successfully. Explain what the output and image reveal in simple terms a beginner can understand. Focus on key insights, trends, or patterns a human would notice. Use real-world analogies if helpful."

        result = eda_explainer.invoke([HumanMessage(content=prompt)])
        explanation_text = result["responses"][0].explanation

        if image_base64:
            display(Image(data=base64.b64decode(image_base64)))

        return {
            "explanation": [explanation_text]
        }
