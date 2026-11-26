import numpy as np
import pytest

from holonomy_dm import channels as ch
from holonomy_dm.protocols import Model, WITNESS, all_words, distance, output_states

RNG = np.random.default_rng(20261126)
NOISY = Model(eps=np.pi / 3, gamma=RNG.uniform(0, 1, 40), delta=RNG.uniform(-0.2, 0.2, 40),
              eta=RNG.uniform(-0.3, 0.3, 40), rho_c=RNG.uniform(0, 1, 40))


@pytest.mark.parametrize("name", "AB")
@pytest.mark.parametrize("h", (1, -1))
def test_every_letter_is_cptp(name, h):
    tp, choi_min = ch.cptp_defects(NOISY.letter(name, h))
    assert tp < 1e-12
    assert choi_min > -1e-12


def test_kraus_and_superoperator_agree():
    k = NOISY.letter("A", 1)
    rho = np.array([[0.3, 0.1 - 0.2j], [0.1 + 0.2j, 0.7]])
    direct = ch.apply(k, rho)
    vec = ch.kraus_to_superop(k) @ rho.reshape(-1, order="F")
    assert np.allclose(direct, np.swapaxes(vec.reshape(-1, 2, 2), -1, -2), atol=1e-13)
    assert np.allclose(ch.apply(ch.superop_to_kraus(ch.kraus_to_superop(k)), rho), direct, atol=1e-12)


def test_dephasing_only_damps_coherences():
    rho = np.array([[0.4, 0.3 + 0.1j], [0.3 - 0.1j, 0.6]])
    out = ch.apply(ch.dephasing_channel(0.37), rho)
    assert np.allclose(np.diag(out), np.diag(rho))
    assert np.isclose(out[0, 1], rho[0, 1] * np.exp(-0.37))


def test_sensing_window_matches_lindblad_solution():
    from scipy.linalg import expm
    gamma, eta = 0.23, 0.07
    for h in (1, -1):
        ham = 0.5 * h * (np.pi / 2) * (1 + eta) * ch.SZ
        s = expm(ch.liouvillian(ham, gamma))
        assert np.allclose(ch.kraus_to_superop(ch.sensing_channel(h, gamma, eta)), s, atol=1e-13)


def test_finite_pulse_reduces_to_ideal_rotation():
    ideal = distance(WITNESS, Model())
    assert abs(distance(WITNESS, Model(rho_c=1e-10)) - ideal) < 1e-8
    k = ch.control_channel(1, 0.9, rho_c=0.4, gamma=0.0, eta=-1.0)   # eta=-1: field off
    u = ch.rot(ch.SY, 0.9)
    assert np.allclose(ch.kraus_to_superop(k), np.kron(u.conj(), u), atol=1e-12)


def test_trace_distance_closed_form_matches_eigenvalues():
    for word in ("AB", "ABBA", WITNESS, "BABABBAAB"):
        plus, minus = output_states(word, NOISY)
        assert np.allclose(ch.trace_distance(plus, minus), ch.trace_distance_fast(plus, minus), atol=1e-13)


def test_hypotheses_are_complex_conjugates_so_sigma_y_readout_is_optimal():
    """rho_- = conj(rho_+) in every noise model here, hence rho_+ - rho_- is
    proportional to sigma_y and a sigma_y measurement attains the Helstrom bound."""
    for word in all_words(7, 1):
        plus, minus = output_states(word, NOISY)
        assert np.allclose(minus, plus.conj(), atol=1e-12)
        diff = plus - minus
        assert np.allclose(diff[..., 0, 0], 0, atol=1e-12)
        assert np.allclose(diff[..., 0, 1].real, 0, atol=1e-12)
        y_plus = np.trace(plus @ ch.SY, axis1=-2, axis2=-1).real
        assert np.allclose(np.abs(y_plus), ch.trace_distance(plus, minus), atol=1e-12)


def test_negative_dephasing_rejected():
    with pytest.raises(ValueError):
        ch.dephasing_channel(-0.1)
