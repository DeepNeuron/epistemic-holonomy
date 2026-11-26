# Independent density-matrix replication, noise sweeps, tests and CI

Independent attempt to falsify the restricted single-qubit discrimination result. New code only; `epistemic_holonomy_certificate.py` is untouched and the README gains one pointer line. Full report: `replication/REPLICATION.md`.

## Outcome

- The headline claims survive. The witness value (D² = 243/256), the absence of any advantage at budgets ≤ 8, its existence at 9, and the grouped ceiling at ε = π/3 are all reproduced with exact arithmetic by an implementation that shares no code or method with the certificate.
- No numerical discrepancy between implementations (worst deviation 4e-14).
- The advantage is narrow once noise is added. Under pure dephasing it vanishes exactly at γ\* = ½ ln(8/7) ≈ 0.0668 per sensing window; no word of up to 12 operations beats `AB` beyond that at ε = π/3.

## Existing certificate, rerun unmodified

Exit 0, output identical to the README's expected lines, 3.8–4.4 s wall (Python 3.13.18, SymPy 1.13.3). Depth-9 line actually printed: `9 56 ['AABABABAB', 'ABABAABAB', 'ABABABAAB']`.

## What is added

- `replication/holonomy_dm`: explicit density matrices, unitaries and Kraus channels; exhaustive unpruned enumeration of all 8191 words to depth 12; exact witness, exact ceiling and an exact universal search (Sturm counting on irreducible factors) to depth 12.
- CPTP dephasing in sensing windows; sweeps over dephasing, pulse-angle error, control duration and field uncertainty (static and shot-to-shot).
- Comparators: fixed-alphabet grouped words (equal budget, equal duration, up to 60 pulses) and an optimised Ramsey protocol.
- Six figures (PDF + PNG), JSON/CSV results, 28 regression tests, pinned requirements and lock file, GitHub Actions workflow.

## Where the advantage holds (ε = π/3)

| Axis | Witness beats grouped ≤ 9 while |
|---|---|
| Dephasing per window | γ < 0.0668 (exact) |
| Known pulse-area error | −3.69% to +3.44% |
| Known field-magnitude error | −16.8% to +20.6% |
| Pulse duration / window | < 0.127 |
| Shot-to-shot field spread | σ < 0.244 |
| Shot-to-shot pulse spread | whole tested range (≤ 0.15) |

## Discrepancies (documented, not repaired)

1. Unique-polynomial counts differ (56 vs 59 at depth 9): signed y over A…B words versus D² over all words. 11 of the certificate's cases are sign duplicates. No effect on verdicts.
2. Two of the three depth-9 words the certificate prints are not witnesses at ε = π/3 (D = 0.758 and 0.325); they work near 2π/3.
3. "Equal counts" is degenerate: the grouped word with the witness's counts has D = 0.
4. README gives no Python version or runtime.
5. The script asserts lengths 2–9 on canonical words only; the rest rests on the README's reduction, which the exhaustive enumeration confirms.

## Limitations

- The unlimited grouped ceiling holds only at exactly ε = π/3: a known 0.49% pulse-area offset lets a grouped word of at most 60 pulses beat the witness.
- Three repeated `AB` shots (6 operations, with resets) already give D = 0.992 > 0.974.
- An optimised Ramsey protocol is ahead or equal in the ideal model, under dephasing and under calibration offsets. In two regions (field spread ≥ 0.21, pulse duration ≥ 0.685 of a window) a fixed-alphabet word beats the one-pulse, one-window Ramsey family used here; that reflects a weak baseline, not an advantage over optimised control.
- Numerical results are grid- and depth-limited evidence, not proofs. No T₁, correlated noise or readout error. No experiment. No claim of general metrological advantage.

## Reproduce

```sh
python -m pip install -r replication/requirements-lock.txt
python epistemic_holonomy_certificate.py
python -m pytest replication/tests
python replication/run_all.py && python replication/make_figures.py
```
