import pandas as pd
import pytest
from mi_agent.nodes.merge import MergeNode
from mi_agent.app_config import settings

@pytest.fixture
def two_csvs(tmp_path):
    df1 = pd.DataFrame({"id":[1,2], "a":[10,20]})
    df2 = pd.DataFrame({"id":[1,2], "b":[100,200]})
    p1 = tmp_path/"one.csv"; df1.to_csv(p1,index=False)
    p2 = tmp_path/"two.csv"; df2.to_csv(p2,index=False)
    return [str(p1), str(p2)]

def test_can_merge_data_false_single(tmp_path):
    state = {"file_paths": [str(tmp_path/"only.csv")], "llm_data_explanations": [], "problem_statement": ""}
    out = MergeNode.can_merge_data(state)
    assert out["can_merge"] is False

def test_merge_data(two_csvs, tmp_path):
    settings.output_dir = str(tmp_path/"out")
    state = {"file_paths": two_csvs, "merge_type": "inner"}
    res = MergeNode.merge_data(state)
    merged = pd.read_csv(res["file_paths"][0])
    assert set(merged.columns) == {"id","a","b"}
    assert merged.shape == (2,3)
