from pathlib import Path


def test_only_oracle_selector_mentions_ground_truth_or_oracle_context():
    root = Path(__file__).parents[1] / "src/sim2act_baselines/active"
    deployable = [
        "random_selector.py", "coverage_nbv_adapted.py", "defect_prior_nbv_adapted.py", "uads_uncertainty_nbv_adapted.py"
    ]
    for name in deployable:
        text = (root / name).read_text().lower()
        assert "ground_truth" not in text
        assert "oraclecontext" not in text
        assert "oracle_context" not in text
    oracle = (root / "oracle_nbv.py").read_text()
    assert "class OracleContext" in oracle
    assert "ground_truth" in oracle
