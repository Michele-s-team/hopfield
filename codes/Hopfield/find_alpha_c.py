import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import root

# ============================================================
# Parameters
# ============================================================

T = 0.2
beta = 1.0 / T

# Number of Gauss-Hermite points
Ngh = 80
x, w = hermgauss(Ngh)

# Transform for standard normal:
# ∫Dz f(z) = 1/√π Σ w_i f(√2 x_i)
z = np.sqrt(2.0) * x
weights = w / np.sqrt(np.pi)


# ============================================================
# Mean-field equations
# ============================================================

def integrals(m, q, alpha):
    """
    Returns (Fm, Fq, r)
    """

    den = 1.0 - beta * (1.0 - q)

    if den <= 0:
        return None

    r = q / den**2

    h = beta * (m + np.sqrt(alpha * r) * z)

    th = np.tanh(h)

    m_new = np.sum(weights * th)
    q_new = np.sum(weights * th**2)

    return m_new, q_new, r


def equations(vars, alpha):
    m, q = vars

    out = integrals(m, q, alpha)

    if out is None:
        return [1e3, 1e3]

    m_new, q_new, r = out

    return [
        m_new - m,
        q_new - q
    ]


# ============================================================
# Solve retrieval state
# ============================================================

def retrieval_exists(alpha, guess=(0.9, 0.9)):

    sol = root(
        equations,
        guess,
        args=(alpha,),
        method='hybr'
    )

    if not sol.success:
        return False, None

    m, q = sol.x

    if m < 1e-3:
        return False, sol.x

    return True, sol.x


# ============================================================
# Binary search for alpha_c
# ============================================================

a_low = 0.10
a_high = 0.20

guess = (0.9, 0.9)

for _ in range(35):

    amid = 0.5 * (a_low + a_high)

    ok, sol = retrieval_exists(amid, guess)

    if ok:
        a_low = amid
        guess = sol
    else:
        a_high = amid

print()
print("Estimated alpha_c =", a_low)