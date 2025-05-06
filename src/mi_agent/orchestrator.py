"""Build and run the InsightForge StateGraph end-to-end."""
import os
import pdfkit
import markdown
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langgraph.checkpoint.memory import MemorySaver
from mi_agent.nodes.file_paths import FilePathNode
from mi_agent.nodes.data_explainer import DataExplainer
from mi_agent.nodes.merge import MergeNode
from mi_agent.nodes.feature_selection import FeatureSelector
from mi_agent.nodes.eda_plan import EDAPlanner
from mi_agent.nodes.eda_executor import EDAExecutor
from mi_agent.nodes.baseline import QuickBaseline
from mi_agent.nodes.tuning import HyperparameterTuner
from mi_agent.nodes.model_explanation import ModelExplainer
from mi_agent.nodes.report import ReportGenerator
from mi_agent.nodes.merge import MergeNode
from mi_agent.states import MIExpertState, EDAExecutionState
from mi_agent.app_config import Settings, settings

def initiate_eda_executions(state: MIExpertState):
    return [
        Send(
            "eda_execution_subgraph",
            {
                "plan": plan,
            },
        )
        for plan in state["eda_plans"]
    ]

def build_graph() -> StateGraph:
    """Instantiate StateGraph, wire nodes & return."""
    builder = StateGraph(MIExpertState)  # your MIExpertState TypedDict
    builder.add_node("identify_file_paths", FilePathNode.identify_file_paths)
    builder.add_node("read_and_explain_data", DataExplainer.read_and_explain_data)
    builder.add_node("can_merge_data", MergeNode.can_merge_data)
    builder.add_node("merge_data", MergeNode.merge_data)
    builder.add_node("select_target_and_features", FeatureSelector.select_target_and_features)
    builder.add_node("generate_eda_plan", EDAPlanner.generate_eda_plan)
    # subgraph
    sub = StateGraph(EDAExecutionState)
    sub.add_node("run_eda_code", EDAExecutor.run_eda_code)
    sub.add_node("explain_eda_result", EDAExecutor.explain_eda_result)
    sub.add_edge(START, "run_eda_code")
    sub.add_edge("run_eda_code", "explain_eda_result")
    sub.add_edge("explain_eda_result", END)
    builder.add_node("eda_execution_subgraph", sub.compile(checkpointer=MemorySaver()), input=None)
    builder.add_node("quick_baseline", QuickBaseline.run_quick_baseline)
    builder.add_node("hyperparameter_tuning", HyperparameterTuner.hyperparameter_tuning)
    builder.add_node("ml_explanation_node", ModelExplainer.ml_explanation_node)
    builder.add_node("executive_summary_node", ReportGenerator.executive_summary_node)

    # wiring
    builder.add_edge(START,       "identify_file_paths")
    builder.add_edge("identify_file_paths","read_and_explain_data")
    builder.add_edge("read_and_explain_data","can_merge_data")
    builder.add_conditional_edges("can_merge_data", MergeNode.should_merge, ["merge_data","select_target_and_features"])
    builder.add_edge("merge_data", "read_and_explain_data")
    builder.add_edge("select_target_and_features","generate_eda_plan")
    builder.add_conditional_edges("generate_eda_plan", initiate_eda_executions, ["eda_execution_subgraph"])
    builder.add_edge("eda_execution_subgraph","quick_baseline")
    builder.add_edge("quick_baseline","hyperparameter_tuning")
    builder.add_edge("hyperparameter_tuning","ml_explanation_node")
    builder.add_edge("ml_explanation_node","executive_summary_node")
    builder.add_edge("executive_summary_node", END)
    return builder.compile(checkpointer=MemorySaver())

def run(
    problem_txt_path: str,
    output_dir: str | None = None,
    model_name: str | None = None,
):
    """
    Run the full MI-Agent pipeline on the given problem statement, then
    render & save the executive_summary.pdf.

    Args:
        problem_txt_path: path to a .txt file containing the problem.
        output_dir:    Optional override of mi_agent.config.OUTPUT_DIR;
                       if None, falls back to the environment‐aware default.
    """
    
    # target_dir = settings.output_dir
    # 1) Collect CLI overrides
    override = {}
    if output_dir:
        override["output_dir"] = output_dir
    if model_name:
        override["model_name"] = model_name
    # 2) Re-create and inject into our singleton
    new_settings = Settings(**override)
    settings.__dict__.update(new_settings.model_dump())

    # now use the (possibly overridden) output_dir
    target_dir = settings.output_dir

    # read problem statement
    with open(problem_txt_path, "r", encoding="utf-8") as f:
        problem = f.read().strip()

    # build & run
    graph = build_graph()
    thread = {"configurable": {"thread_id": "final_eda_workflow"}}
    inputs = {"problem_statement": problem}
    for _ in graph.stream(inputs, thread, stream_mode="values"):
        pass

    # generate PDF
    final_state = graph.get_state(thread).values
    md = final_state["executive_summary"]
    html_body = markdown.markdown(md)
    html = f"""<!DOCTYPE html>
    <html lang="en"><head><meta charset="utf-8">
    <style>
    body {{ font-family: Arial, sans-serif; margin:1in; line-height:1.4; }}
    h1,h2,h3 {{ page-break-after: avoid; }}
    p,li {{ orphans:3; widows:3; }}
    ul {{ margin-top:0; }}
    </style>
    </head><body>
    {html_body}
    </body></html>
    """

    opts = {
      "page-size": "A4",
      "margin-top": "0.5in",
      "margin-bottom": "0.5in",
      "margin-left": "0.3in",
      "margin-right": "0.3in",
      "encoding": "UTF-8",
      "enable-local-file-access": "",
    }

    os.makedirs(target_dir, exist_ok=True)
    pdf_path = os.path.join(target_dir, "executive_summary.pdf")
    pdfkit.from_string(html, pdf_path, options=opts)
    print(f"✅ Saved nicely formatted PDF to {pdf_path}")
