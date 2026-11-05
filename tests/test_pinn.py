import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import synthetic_unit_piml_pinn as S


@pytest.fixture(scope="module")
def scc_df():
    return S.simulate_scc_alternatives(100)


@pytest.fixture(autouse=True)
def checkpoint_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT_PINN", tmp_path / "pinn.pt")


def test_deltaK_hand_value():
    # 1000 kPag = 1 MPa, 8 in = 0.2032 m: 12 * 1 * (0.2032 / 0.2) = 12.192
    got = S.deltaK_surrogate(torch.tensor(1000.0), torch.tensor(8.0))
    assert float(got) == pytest.approx(12.192, rel=1e-5)


def test_training_feeds_deltaK_physical_units(scc_df, monkeypatch):
    seen = []
    real = S.deltaK_surrogate
    monkeypatch.setattr(S, "deltaK_surrogate", lambda op, dn: seen.append(op) or real(op, dn))
    S.train_pinn_demo(scc_df, epochs=1)
    # make_pipe_attrs draws operating pressure from 50-2500 kPag
    assert float(seen[0].min()) >= 50.0


def test_crack_grows_after_training(scc_df, monkeypatch):
    last = {}
    real = S.simulate_crack_growth_torch

    def spy(*args, **kwargs):
        last["traj"] = real(*args, **kwargs)
        return last["traj"]

    monkeypatch.setattr(S, "simulate_crack_growth_torch", spy)
    S.train_pinn_demo(scc_df)
    assert float(last["traj"][:, -1].median()) > 2 * float(last["traj"][:, 0].median())


def test_training_is_reproducible(scc_df):
    def weights():
        S.train_pinn_demo(scc_df, epochs=2)
        return torch.load(S.OUT_PINN, weights_only=False)["state_dict"]

    a, b = weights(), weights()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_heldout_susceptibility_beats_constant(scc_df):
    m = S.train_pinn_demo(scc_df)["metrics"]
    assert m["susceptibility_rmse"] < m["susceptibility_rmse_constant"]
