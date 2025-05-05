# tests/test_eda_executor.py

import os
import pytest
from mi_agent.nodes.eda_executor import EDAExecutor
import mi_agent.app_config as app_cfg

class DummyResp:
    def __init__(self, code: str):
        self.code = code

@pytest.fixture(autouse=True)
def stub_deps(monkeypatch, tmp_path):
    # Redirect output_dir
    app_cfg.settings.output_dir = str(tmp_path / "out")

    import mi_agent.nodes.eda_executor as mod

    # 1) Stub eda_code_extractor.invoke to return code that writes an image
    class FakeExtractor:
        @staticmethod
        def invoke(prompt):
            # code that will write 'plan1.png' in the cwd
            return {"responses": [DummyResp("with open('plan1.png','wb') as f: f.write(b'dummy')")]}

    monkeypatch.setattr(mod, "eda_code_extractor", FakeExtractor)

    # 2) Stub PythonREPL.run to execute our fake code and return "OK"
    class FakeREPL:
        def run(self, code):
            # execute the code string so it writes plan1.png
            exec(code, {})
            return "OK"

    monkeypatch.setattr(mod, "PythonREPL", FakeREPL)

    # 3) Patch move_file **inside** the eda_executor module to mimic moving a file
    def fake_move(src, dst):
        os.makedirs(dst, exist_ok=True)
        target = os.path.join(dst, os.path.basename(src))
        os.replace(src, target)
        return target

    monkeypatch.setattr(mod, "move_file", fake_move)

    # 4) Stub the explainer so explain_eda_result can run
    class FakeExplainer:
        @staticmethod
        def invoke(msgs):
            return {"responses": [type("R", (), {"explanation": "Good explanation"})]}

    monkeypatch.setattr(mod, "eda_explainer", FakeExplainer)

def test_run_eda_code(tmp_path):
    cfg = {
        "eda_plan_name": "plan1",
        "eda_plan_description": "desc",
        "problem_statement": "ps",
        "file_paths": [],
        "target_column": "t",
        "feature_columns": []
    }
    state = {"plan": type("P", (), {"execution_state": cfg}), "max_retries": 1}
    out = EDAExecutor.run_eda_code(state)

    assert "generated_code" in out
    assert out["code_output"] == "OK"
    # Now image_path is a string pointing to the moved file
    assert isinstance(out["image_path"], str)
    assert os.path.exists(out["image_path"])

def test_explain_eda_result(tmp_path):
    # Prepare a dummy image under output_dir
    odir = tmp_path / "out"
    odir.mkdir(parents=True, exist_ok=True)
    img = odir / "plan1.png"
    img.write_bytes(b"dummy")

    cfg = {
        "eda_plan_name": "plan1",
        "eda_plan_description": "desc",
        "problem_statement": "ps",
        "file_paths": [],
        "target_column": "t",
        "feature_columns": []
    }
    state = {
        "plan": type("P", (), {"execution_state": cfg}),
        "generated_code": "code",
        "code_output": "OK",
        # explain_eda_result expects a list, but it only uses the first element
        "image_path": [str(img)]
    }
    out = EDAExecutor.explain_eda_result(state)
    assert out["explanation"] == ["Good explanation"]
