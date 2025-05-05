"""Node: identify which CSVs to load."""
from mi_agent.extractors import file_path_extractor
from langchain_core.messages import HumanMessage
from mi_agent.states import GenerateEDAAgentsState

class FilePathNode:
    """Extract file paths from a problem statement."""

    @staticmethod
    def identify_file_paths(state: GenerateEDAAgentsState) -> dict:
        """Ask the LLM which CSV paths to load."""
        prompt = f"""
        You are a helpful assistant. Given the problem: \"{state['problem_statement']}\",
        return a list of dataset file paths (as strings) that should be loaded to help with the analysis.
        """
        result = file_path_extractor.invoke(prompt)
        responses = result.get('responses', [])
        if not responses:
            raise ValueError("No responses found in Trustcall extractor result.")
        file_paths = [f.path for f in responses[0].file_paths]
        return {"file_paths": file_paths}
