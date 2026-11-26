import sympy as sp

from holonomy_dm import exact


def test_witness_exact():
    w = exact.witness_exact()
    assert w["D_squared"] == sp.Rational(243, 256)
    assert sp.simplify(w["D"] - 9 * sp.sqrt(3) / 16) == 0
    assert w["purity_plus"] == 1
    assert w["difference"][0, 0] == 0 and w["difference"][1, 1] == 0
    assert sp.simplify(w["difference_over_sigma_y"] - 9 * sp.sqrt(3) / 16) == 0


def test_unlimited_grouped_ceiling_exact():
    c = exact.grouped_ceiling_exact()
    assert c["A_period"] == 6 and c["ceiling_D_squared"] == sp.Rational(3, 4)


def test_witness_under_dephasing_polynomial():
    lam, poly = exact.witness_dephased_polynomial()
    assert sp.expand(poly - 3 * lam ** 2 * (8 * lam ** 2 + 1) ** 2 / 256) == 0
    # equality with grouped AB, D^2 = 3 lam^2 / 4, exactly at lam^2 = 7/8
    assert sp.expand(poly - sp.Rational(3, 4) * lam ** 2).subs(lam, sp.sqrt(sp.Rational(7, 8))) == 0


def test_sturm_decision_procedure_on_known_cases():
    t = sp.symbols("t")
    P = lambda e: sp.Poly(e, t, domain="QQ")
    assert exact.strictly_feasible(P(1 - t ** 2), [P(0)])[0]
    assert not exact.strictly_feasible(P(t ** 2), [P(t ** 2)])[0]
    assert not exact.strictly_feasible(P(t ** 2 - 2), [P(0)])[0]
    # feasible only on a tiny window between two close irrational roots
    assert exact.strictly_feasible(P(-(t ** 2 - sp.Rational(1, 2)) * (t ** 2 - sp.Rational(5001, 10000))), [P(0)])[0]
    # touching (double root) is not strict
    assert not exact.strictly_feasible(P(-(3 * t - 1) ** 2), [P(0)])[0]
    # two competitors that are each beaten somewhere but never simultaneously
    assert not exact.strictly_feasible(P(0), [P(t), P(-t)])[0]


def test_universal_search_matches_claim_through_depth_9():
    report = exact.universal_search(9)
    for length in range(0, 9):
        assert report[length]["witnesses"] == []
    assert report[9]["witnesses"] == ["AABABABAB", "ABABAABAB", "ABABABAAB"]
    assert report[9]["unique_D2_polynomials"] == 59
