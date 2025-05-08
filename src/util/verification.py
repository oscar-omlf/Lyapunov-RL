# verification.py
import numpy as np
import dreal as d

def dreal_var(n, prefix='x'):
    return np.array([d.Variable(f'{prefix}{i}') for i in range(n)])

def lie_derivative(expr, xvars, f_cl):
    return sum(expr.Differentiate(xi) * fi for xi, fi in zip(xvars, f_cl))

def in_R2(x, lb, ub, scale=1.):
    terms = []
    for i in range(len(x)):
        terms += [x[i] >= lb[i]*scale, x[i] <= ub[i]*scale]
    return d.And(*terms)

def boundary_R2(x, lb, ub, scale=2., tol=0.03):
    inner = in_R2(x, lb*(scale*(1-tol)), ub*(scale*(1-tol)))
    outer = in_R2(x, lb*scale,            ub*scale)
    return d.And(outer, d.Not(inner))

def closed_loop_f_dreal(dynamics_dreal, actor, x):
    u = actor.model.forward_dreal(x)        # shape (1,) or (nu,)
    if not isinstance(u, np.ndarray):
        u = np.array([u])
    return dynamics_dreal(x, u)             # np.array of dReal Expr

def verify_W(critic, actor, dynamics_dreal,
             lb, ub, c, eps=0.05, delta=1e-4, scale=2.):
    nx = len(lb)
    x = dreal_var(nx)

    Wx   = critic.forward_dreal(x)[0]
    W0   = critic.forward_dreal(np.zeros(nx))[0]

    f_cl = closed_loop_f_dreal(dynamics_dreal, actor, x)
    lieW = lie_derivative(Wx, x, f_cl)

    xnorm = d.sqrt(sum(xi*xi for xi in x))

    cond1 = d.And(in_R2(x, lb, ub, scale),
                  Wx < c,
                  xnorm >= eps,
                  Wx <= W0)
    cond2 = d.And(in_R2(x, lb, ub, scale),
                  Wx < c,
                  xnorm >= eps,
                  lieW >= 0)
    cond3 = d.And(boundary_R2(x, lb, ub, scale),
                  Wx <= c)

    unsat1 = d.CheckSatisfiability(cond1, delta).is_false()
    unsat2 = d.CheckSatisfiability(cond2, delta).is_false()
    unsat3 = d.CheckSatisfiability(cond3, delta).is_false()
    return unsat1 and unsat2 and unsat3

def bisect_c_ac(critic, actor, dynamics_dreal,
                lb, ub, eps=0.05, delta=1e-4, scale=2., tol=1e-3):
    lo, hi = 0.0, 0.99           # sigmoid ⇒ W∈(0,1)
    while hi - lo > tol:
        mid = 0.5*(lo+hi)
        ok = verify_W(critic, actor, dynamics_dreal,
                      lb, ub, mid, eps, delta, scale)
        (lo, hi) = (mid, hi) if ok else (lo, mid)
    return lo


def verify_V_lqr(P, K, dynamics_dreal, lb, ub,
                 rho, eps=0.05, delta=1e-4, scale=2.):
    """
    P : (n,n) positive-definite   (solution of CARE)
    K : (m,n) state-feedback gain
    We prove  V(x)=xᵀPx  is Lyapunov on {x | V(x) < rho}.
    """
    nx = len(lb)
    x = dreal_var(nx)

    Vx = sum(sum(x[i]*P[i,j]*x[j] for j in range(nx)) for i in range(nx))
    u = -np.dot(K, x)            # algebraic u = -Kx (np.ndarray of Expr)
    u = np.array([u])

    f_cl = dynamics_dreal(x, u)

    lieV = lie_derivative(Vx, x, f_cl)
    xnorm = d.sqrt(sum(xi*xi for xi in x))

    cond1 = d.And(in_R2(x, lb, ub, scale),
                  Vx < rho,
                  xnorm >= eps,
                  lieV >= 0)         # negate “< 0”

    cond2 = d.And(boundary_R2(x, lb, ub, scale),
                  Vx <= rho)         # should be > rho

    unsat1 = d.CheckSatisfiability(cond1, delta).is_false()
    unsat2 = d.CheckSatisfiability(cond2, delta).is_false()
    return unsat1 and unsat2


def bisect_rho_lqr(P, K, dynamics_dreal,
                   lb, ub, eps=0.05, delta=1e-4,
                   scale=2., tol=1e-2, rho_max=20.):
    lo, hi = 0.0, rho_max
    while hi - lo > tol:
        mid = 0.5*(lo+hi)
        ok = verify_V_lqr(P, K, dynamics_dreal,
                          lb, ub, mid, eps, delta, scale)
        (lo, hi) = (mid, hi) if ok else (lo, mid)
    return lo
