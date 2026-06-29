"""tests/test_smoke.py - Smoke tests for UpliftIQ pipeline."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_make_synthetic():
    from src.data import make_synthetic_uplift
    data = make_synthetic_uplift(n=1000)
    assert data["X"].shape == (1000, 10)
    assert data["y"].shape == (1000,)
    assert data["treatment"].shape == (1000,)
    assert len(np.unique(data["treatment"])) == 3


def test_compute_ate():
    from src.data import make_synthetic_uplift
    from src.model import compute_ate
    data = make_synthetic_uplift(n=5000)
    # Use only arm 0 and 1
    mask = np.isin(data["treatment"], [0, 1])
    ate = compute_ate(data["y"][mask], data["treatment"][mask])
    assert "ate" in ate
    assert "ci_lower" in ate
    assert "ci_upper" in ate
    assert ate["ci_lower"] < ate["ate"] < ate["ci_upper"]


def test_t_learner():
    from src.data import make_synthetic_uplift, prepare_binary_uplift
    from src.model import TLearner
    data = make_synthetic_uplift(n=2000)
    binary = prepare_binary_uplift(data, treatment_col=1)
    model = TLearner()
    model.fit(binary["X"], binary["treatment"], binary["y"])
    cate = model.predict_cate(binary["X"][:100])
    assert cate.shape == (100,)


def test_s_learner():
    from src.data import make_synthetic_uplift, prepare_binary_uplift
    from src.model import SLearner
    data = make_synthetic_uplift(n=2000)
    binary = prepare_binary_uplift(data, treatment_col=1)
    model = SLearner()
    model.fit(binary["X"], binary["treatment"], binary["y"])
    cate = model.predict_cate(binary["X"][:100])
    assert cate.shape == (100,)


def test_qini():
    from src.data import make_synthetic_uplift, prepare_binary_uplift
    from src.model import TLearner, compute_qini
    data = make_synthetic_uplift(n=2000)
    binary = prepare_binary_uplift(data, treatment_col=1)
    model = TLearner()
    model.fit(binary["X"], binary["treatment"], binary["y"])
    cate = model.predict_cate(binary["X"])
    fractions, curve, qini = compute_qini(binary["y"], binary["treatment"], cate)
    assert len(fractions) == 100
    assert isinstance(qini, float)


if __name__ == "__main__":
    test_make_synthetic()
    test_compute_ate()
    test_t_learner()
    test_s_learner()
    test_qini()
    print("All smoke tests passed!")
