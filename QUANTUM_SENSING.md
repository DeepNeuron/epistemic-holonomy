# Binary quantum sensing: an idealized magnetometry protocol

This note interprets the existing restricted-control theorem, not an experimental demonstration.

Let a spin-1/2 probe experience an unknown signed field h B0 along z, h=+/-1, with gyromagnetic sign absorbed into h. During a sensing window tau, H_h=h hbar omega0 sigma_z/2, omega0 tau=pi/2. B_h is the resulting Bloch rotation. A is a calibrated y pulse of area pi/3, assumed independent of h; the ideal model neglects the unknown field during control pulses. Prepare +Z and apply ABABABAAB chronologically (five control pulses, four sensing windows). Measure sigma_y. The two hypotheses are equally likely and the y readout is Helstrom optimal by reflection symmetry.

The exact witness gives D=9 sqrt(3)/16 and single-shot success P=(1+D)/2, about 98.71%. The best single-shot grouped sequence, even allowing arbitrarily many ideal traversals, gives D=sqrt(3)/2 and P about 93.30%. Corresponding error probabilities are about 1.29% and 6.70%. This is not a bound on repeated independent grouped experiments or arbitrary pulse areas, initial states, adaptive measurements or optimized controls. A grouped Ramsey experiment with freely chosen pi/2 preparation can perfectly distinguish these ideal hypotheses, so restriction to the fixed gate alphabet is essential.

## Next experiments

- Implement the two sign hypotheses on a controllable spin, e.g. an NV center or a qubit emulator. Use fixed calibrated pulse areas and randomized blinded sign labels.
- Compare same total sensing exposure, control duration, state preparation/readout overhead and repetition count; report total error and confidence intervals.
- Replace ideal rotations by channels with T1/T2 relaxation, pulse errors and field evolution during control. Map where the advantage disappears.
- For continuous field estimation replace B_h by B(theta)=R_z(theta). Compute classical Fisher information for actual measurement probabilities and quantum Fisher information for the output states. Compare against optimized Ramsey and constrained optimal control at equal total time. Binary trace distance does not establish local Fisher-information advantage.
- Report nuisance-adjusted Fisher information when pulse calibration, detuning or readout contrast are unknown; use a Schur complement when the nuisance block is invertible and address singular cases explicitly.

## Prior art and novelty boundary

Coherent control as a means to improve quantum metrology is established. The specific finite-word theorem and sharp depth under its fixed alphabet are the candidate contributions; novelty has not been established by a comprehensive literature review.

- Pang and Jordan, Optimal adaptive control for quantum metrology with time-dependent Hamiltonians, Nature Communications 8, 14695 (2017), https://doi.org/10.1038/ncomms14695
- Liu and Yuan, Quantum parameter estimation with optimal control, Physical Review A 96, 012117 (2017), https://doi.org/10.1103/PhysRevA.96.012117
- Yang et al., Variational Principle for Optimal Quantum Controls in Quantum Metrology, Physical Review Letters 128, 160505 (2022), https://doi.org/10.1103/PhysRevLett.128.160505
