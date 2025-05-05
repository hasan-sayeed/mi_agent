import pytest
from mi_agent.nodes.file_paths import FilePathNode

class DummyResp:
    def __init__(self, paths):
        # each response must have a .file_paths list of objects with a .path attribute
        self.file_paths = [type("FP", (), {"path": p}) for p in paths]

class FakeExtractor:
    def invoke(self, prompt):
        return {"responses": [DummyResp(["data/foo.csv", "data/bar.csv"])]}

@pytest.fixture(autouse=True)
def stub_extractor(monkeypatch):
    import mi_agent.nodes.file_paths as mod
    # entirely replace the extractor with our fake
    monkeypatch.setattr(mod, "file_path_extractor", FakeExtractor())

def test_identify_file_paths():
    state = {"problem_statement": "some problem"}
    out = FilePathNode.identify_file_paths(state)
    assert out["file_paths"] == ["data/foo.csv", "data/bar.csv"]
