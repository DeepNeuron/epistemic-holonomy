"""Explicit 2x2 unitaries and CPTP maps in Kraus form.

Conventions
-----------
* Computational basis |0>, |1>; sigma_z|0> = +|0>; the +Z Bloch state is |0><0|.
* R_n(theta) = exp(-i theta n.sigma / 2): an active right-handed Bloch rotation.
* A channel is an array of Kraus operators with shape (..., K, 2, 2); leading
  axes broadcast over parameter grids.  rho -> sum_k K_k rho K_k^dagger.
* Hypothesis h = +1 / -1 is the sign of the field along z.

Noise parameters (all dimensionless)
------------------------------------
gamma : dephasing exponent per sensing window; coherences are multiplied by
        exp(-gamma) in each window (gamma = tau / T2).
delta : fractional systematic pulse-area error, A = R_y(eps (1 + delta)).
eta   : fractional field-magnitude error, sensing angle h (pi/2)(1 + eta).
rho_c : control duration in units of the sensing window, t_pulse / tau.
        For rho_c > 0 the unknown field (and dephasing at the same rate)
        acts during the control pulse.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
RHO0 = np.array([[1, 0], [0, 0]], dtype=complex)


def rot(axis: np.ndarray, theta) -> np.ndarray:
    """exp(-i theta axis/2) for a Pauli ``axis``; theta may be an array."""
    theta = np.asarray(theta, dtype=float)[..., None, None]
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * axis


def unitary_channel(u: np.ndarray) -> np.ndarray:
    return np.asarray(u, dtype=complex)[..., None, :, :]


def compose(second: np.ndarray, first: np.ndarray) -> np.ndarray:
    """Kraus operators of (second o first): all products K2_j K1_k."""
    prod = np.einsum("...jab,...kbc->...jkac", second, first)
    return prod.reshape(prod.shape[:-4] + (-1, 2, 2))


def dephasing_channel(gamma) -> np.ndarray:
    """Phase-flip channel with Kraus {sqrt(1-p) I, sqrt(p) Z}, 1-2p = exp(-gamma)."""
    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma < 0):
        raise ValueError("gamma must be non-negative")
    p = (1.0 - np.exp(-gamma)) / 2.0
    k0 = np.sqrt(1 - p)[..., None, None] * I2
    k1 = np.sqrt(p)[..., None, None] * SZ
    return np.stack([k0, k1], axis=-3)


def sensing_channel(h: int, gamma=0.0, eta=0.0, duration=1.0) -> np.ndarray:
    """Free evolution under H = h omega0 sigma_z/2 for ``duration`` windows,
    with pure dephasing.  Unitary and dephasing commute, so this equals the
    exact solution of the Lindblad equation for the window."""
    gamma, eta, duration = np.broadcast_arrays(
        np.asarray(gamma, float), np.asarray(eta, float), np.asarray(duration, float))
    u = rot(SZ, h * (np.pi / 2) * (1 + eta) * duration)
    return compose(dephasing_channel(gamma * duration), unitary_channel(u))


def liouvillian(hamiltonian: np.ndarray, dephase_rate) -> np.ndarray:
    """Column-stacking superoperator of d rho/dt = -i[H,rho] + (r/2)(Z rho Z - rho)."""
    h = np.asarray(hamiltonian, dtype=complex)
    ht = np.swapaxes(h, -1, -2)
    r = np.asarray(dephase_rate, dtype=float)[..., None, None]
    return (-1j * (np.kron(I2, h) - np.kron(ht, I2))
            + (r / 2) * (np.kron(SZ, SZ) - np.eye(4)))


def superop_to_kraus(s: np.ndarray, tol: float = 1e-13) -> np.ndarray:
    """Convert a column-stacking superoperator to four Kraus operators through
    the Choi matrix.  Raises if the Choi matrix is not positive semidefinite."""
    s = np.asarray(s, dtype=complex)
    lead = s.shape[:-2]
    # S[(a,b),(c,d)] with vec index = row + 2*col: S[a+2b, c+2d] = <a|E(|c><d|)|b>
    s4 = s.reshape(lead + (2, 2, 2, 2))            # [b, a, d, c]
    choi = np.einsum("...badc->...cadb", s4).reshape(lead + (4, 4))  # [(c,a),(d,b)]
    w, v = np.linalg.eigh(choi)
    if np.min(w) < -1e-9:
        raise ValueError(f"map is not completely positive (min Choi eig {np.min(w)})")
    w = np.clip(w, 0.0, None)
    vecs = v * np.sqrt(w)[..., None, :]            # columns are sqrt(w) v
    # eigenvector index (c,a) -> K[a,c]
    k = np.swapaxes(vecs, -1, -2).reshape(lead + (4, 2, 2))   # [k, c, a]
    return np.swapaxes(k, -1, -2)


def control_channel(h: int, eps, delta=0.0, rho_c=0.0, gamma=0.0, eta=0.0,
                    area_scale=1.0) -> np.ndarray:
    """Control pulse of nominal area eps*area_scale about y.

    rho_c == 0 everywhere: ideal instantaneous R_y (one Kraus operator).
    Otherwise the pulse lasts rho_c*area_scale sensing windows, during which
    the field h (pi/2)(1+eta) per window and dephasing gamma per window act
    simultaneously with the drive; the map is exp(Liouvillian), converted to
    Kraus form.
    """
    eps, delta, rho_c, gamma, eta, area_scale = np.broadcast_arrays(*[
        np.asarray(x, float) for x in (eps, delta, rho_c, gamma, eta, area_scale)])
    area = eps * (1 + delta) * area_scale
    if not np.any(rho_c):
        return unitary_channel(rot(SY, area))
    dur = rho_c * area_scale
    phi = h * (np.pi / 2) * (1 + eta) * dur
    ham = 0.5 * (area[..., None, None] * SY + phi[..., None, None] * SZ)
    return superop_to_kraus(expm(liouvillian(ham, gamma * dur)))


def apply(channel: np.ndarray, rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho)[..., None, :, :]
    return np.sum(channel @ rho @ np.swapaxes(channel.conj(), -1, -2), axis=-3)


def kraus_to_superop(channel: np.ndarray) -> np.ndarray:
    """Column-stacking superoperator sum_k conj(K) (x) K."""
    return np.einsum("...kbd,...kac->...badc", channel.conj(), channel).reshape(
        channel.shape[:-3] + (4, 4))


def cptp_defects(channel: np.ndarray) -> tuple[float, float]:
    """(max |sum K^dag K - I|, most negative Choi eigenvalue) over all leading axes."""
    tp = np.einsum("...kba,...kbc->...ac", channel.conj(), channel) - I2
    s = kraus_to_superop(channel)
    lead = s.shape[:-2]
    choi = np.einsum("...badc->...cadb", s.reshape(lead + (2, 2, 2, 2))).reshape(lead + (4, 4))
    return float(np.max(np.abs(tp))), float(np.min(np.linalg.eigvalsh(choi)))


def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> np.ndarray:
    """Reference implementation: half the sum of |eigenvalues| of the difference."""
    d = rho_a - rho_b
    d = (d + np.swapaxes(d.conj(), -1, -2)) / 2
    return 0.5 * np.sum(np.abs(np.linalg.eigvalsh(d)), axis=-1)


def trace_distance_fast(rho_a: np.ndarray, rho_b: np.ndarray) -> np.ndarray:
    """Qubit closed form sqrt(-det) for a traceless Hermitian difference."""
    d = rho_a - rho_b
    return np.sqrt(d[..., 0, 0].real ** 2 + np.abs(d[..., 0, 1]) ** 2)
