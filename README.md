# Epistemic Holonomy: exact quantum interleaving certificates

Research-stage mathematical results for binary discrimination with restricted coherent controls. Not peer reviewed.

## Model and theorem

Prepare one qubit in the +Z Bloch state. Under hypotheses h=+/- the available operations are A=R_y(epsilon) and B_h=R_z(h*pi/2), using active right-handed Bloch rotations. Words are chronological; each letter costs one traversal. No intermediate measurements, resets, ancillas or arbitrary additional controls are allowed. The terminal measurement is unrestricted. Define D(w)=||rho_+(w)-rho_-(w)||_1/2. A grouped word has at most one switch between A and B.

**Quarter-turn minimality claim.** For every real epsilon and every budget L<=8, no word of length <=L strictly exceeds the best grouped word of length <=L. At L=9 a strict advantage exists. The supplied symbolic search is intended to certify this universal finite-depth claim via real-root isolation, not a sampled angle grid.

**Explicit witness (including unlimited grouped comparison).** At epsilon=pi/3, w=ABABABAAB has D=9*sqrt(3)/16 (D^2=243/256). Every grouped word of any length has D<=sqrt(3)/2 (D^2<=3/4). Indeed B-first words have zero distance; A^k B^m has D=|sin(k epsilon) sin(m pi/2)|. The maximum follows from periodicity at pi/3. This is a restricted-control result, not an optimization over arbitrary quantum protocols.

For the quarter-turn case the two output Bloch vectors are related by reflection of y. Set t=cos(epsilon), and write the plus vector as (sin(epsilon)x,sin(epsilon)y,z). The polynomial recurrence in the certificate computes y; D=|sin(epsilon)y|. Grouped competitors within budget L have scaled scores U_(k-1)(t), 1<=k<=L-1, where U is the Chebyshev polynomial of the second kind. Leading B and trailing A can be removed without changing distance. Comparing squared scores reduces strict advantage to simultaneous polynomial inequalities on (-1,1); the endpoints give zero distance. This justifies the search reduction.

The certificate also checks two finite-budget counterexamples with changed axes and angles (BAB and ABAB). The broader odd-denominator infinite-family and asymptotic-depth claims are not certified by this repository version.

## Reproduce

Use Python 3 with SymPy. From the repository root:

```sh
python -m pip install sympy==1.13.3
python epistemic_holonomy_certificate.py
```

Do not use python -O (assertions must remain enabled). The search prints per-depth witness lists; lengths 2 through 8 must have empty lists and length 9 a nonempty list. The tilted tests report squared distances 27/32 vs 3/8 and 249/256 vs 3/4. The final expected lines are:

```text
PASS: nine-step witness D^2=243/256; unlimited grouped ceiling D^2=3/4
PASS: all certificate assertions completed
```

These are expected outputs, not a recorded execution log. Independent replication is requested. Cite a specific commit rather than the moving main branch.

## Quantum sensing and metrology

See [the sensing research note](QUANTUM_SENSING.md). The explicit witness concerns binary field-sign discrimination, not local estimation precision, a demonstrated hardware advantage, or a violation of a quantum limit.
