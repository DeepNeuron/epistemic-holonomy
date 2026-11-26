"""Exact finite-depth certificates for the two-loop discrimination problem.

Run with Python 3 and SymPy: python epistemic_holonomy_certificate.py
Words are chronological. The input is +Z and distance is trace distance.
"""
from itertools import product
import sympy as sp

t = sp.symbols("t", real=True)


def scaled_y(word):
    """Bloch vector components are (sin(epsilon)*x, sin(epsilon)*y, z)."""
    x, y, z = 0, 0, 1
    for letter in word:
        if letter == "A":
            x, z = sp.expand(t*x + z), sp.expand(t*z - (1-t**2)*x)
        else:
            x, y = -y, x
    return y


def simultaneous_positive(polynomials):
    """Test strict feasibility in (-1,1) by simultaneous real-root isolation."""
    if any(p.is_zero for p in polynomials):
        return False
    roots = sp.polys.polytools.intervals(
        [p.as_expr() for p in polynomials], inf=-1, sup=1,
        eps=sp.Rational(1, 10**12),
    )
    # Remove roots at the boundaries. Every other root must be isolated
    # strictly inside the domain, with nonempty rational gaps on either side.
    boxes = [(a, b) for (a, b), _ in roots if (a, b) not in [(-1, -1), (1, 1)]]
    left = sp.Integer(-1)
    for a, b in boxes + [(sp.Integer(1), sp.Integer(1))]:
        assert left < a
        sample = (left + a)/2
        if all(p.eval(sample) > 0 for p in polynomials):
            return True
        left = b
    return False


def quarter_turn_search():
    for length in range(2, 10):
        grouped = [sp.chebyshevu(k-1, t) for k in range(1, length)]
        seen = set()
        witnesses = []
        # A leading B fixes +Z. A trailing A preserves trace distance.
        for middle in product("AB", repeat=length-2):
            word = "A" + "".join(middle) + "B"
            score = scaled_y(word)
            if score in seen:
                continue
            seen.add(score)
            differences = [sp.Poly(score**2 - g**2, t) for g in grouped]
            if simultaneous_positive(differences):
                witnesses.append(word)
        print(length, len(seen), witnesses)
        assert bool(witnesses) == (length == 9)
    assert scaled_y("ABABABAAB").subs(t, sp.Rational(1, 2)) == sp.Rational(9, 8)


def rotation(axis, cosine, sine):
    x, y, z = axis
    cross = sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return cosine*sp.eye(3) + (1-cosine)*axis*axis.T + sine*cross


def squared_distance(word, a, b):
    plus = minus = sp.eye(3)[:, 2]
    for letter in word:
        plus = (a if letter == "A" else b)*plus
        minus = (a if letter == "A" else b.T)*minus
    difference = plus-minus
    return sp.simplify(difference.dot(difference)/4)


def verify_tilted(a, b, word, expected, ceiling):
    values = []
    for length in range(len(word)+1):
        for letters in product("AB", repeat=length):
            switches = sum(u != v for u, v in zip(letters, letters[1:]))
            if switches <= 1:
                values.append(squared_distance(letters, a, b))
    assert max(values) == ceiling
    assert squared_distance(word, a, b) == expected > ceiling
    print(word, expected, "grouped ceiling squared", ceiling)


if __name__ == "__main__":
    quarter_turn_search()
    y_axis, z_axis = sp.eye(3)[:, 1], sp.eye(3)[:, 2]
    tilted = (y_axis + z_axis)/sp.sqrt(2)
    half, sine = sp.Rational(1, 2), sp.sqrt(3)/2
    verify_tilted(rotation(y_axis, -1, 0), rotation(tilted, half, sine),
                  "BAB", sp.Rational(27, 32), sp.Rational(3, 8))
    verify_tilted(rotation(tilted, 0, 1), rotation(z_axis, half, sine),
                  "ABAB", sp.Rational(249, 256), sp.Rational(3, 4))
