import numpy as np
import dreal as d


from util.dreal import dreal_var, in_box, on_boundary, is_unsat
from agents.agent_factory import AgentFactory 
from config import config_lac_pendulum, config_lqr_pendulum


LEVEL_INIT = 0.95
MAX_ACTION = 1.0

def bisection(check_fun, c_max=0.95, tol=1e-3, it_max=40, eps=0.5):
    hi_fail = c_max
    while True:
        r1, r2 = check_fun(hi_fail, eps=eps)
        if is_unsat(r1) and is_unsat(r2):
            break
        hi_fail *= 0.5
        if hi_fail < 1e-6:
            print("No certificate even at c = 0.")
            return 0.0
        print(hi_fail)

    lo_pass = hi_fail
    hi_fail = hi_fail * 2.0

    for _ in range(it_max):
        if hi_fail - lo_pass < tol:
            break
        mid = 0.5 * (lo_pass + hi_fail)
        r1, r2 = check_fun(mid, eps=eps)
        ok = is_unsat(r1) and is_unsat(r2)
        print(f"c={mid:.5f}  ok={ok}")
        if ok:
            lo_pass = mid
        else:
            hi_fail = mid
    return lo_pass


CFG_LAC = config_lac_pendulum
CFG_LQR = config_lqr_pendulum

dynamics_fn_dreal = CFG_LAC["dynamics_fn_dreal"]

agent_lac = AgentFactory.create_agent(config=config_lac_pendulum)
agent_lqr = AgentFactory.create_agent(config=CFG_LQR)

agent_lac.load(file_path='best_models/LAC/', episode=0)

agent_lac.trainer._sanity_network()
agent_lac.trainer._sanity_dynamics()

c_star = bisection(agent_lac.trainer.check_lyapunov, LEVEL_INIT)
print(f"\nLyAC certified c* = {c_star:.4f}")

alpha = CFG_LAC["alpha"]
P     = agent_lqr.P_np
R1_LB, R1_UB = CFG_LAC["r1_bounds"]
lb    = R1_LB
ub    = R1_UB

print(f"Alpha: {alpha}, R1_LB: {lb}, R1_UB: {ub}")  


def lqr_check_discrete(level, scale=2., eps=0.5, delta=1e-3):
    n = agent_lqr.P_np.shape[0]
    x = dreal_var(n)

    P = agent_lqr.P_np
    V = sum(x[i] * sum(P[i, j] * x[j] for j in range(n)) for i in range(n))

    K = agent_lqr.K_np
    m = K.shape[0]
    u_lqr = [-sum(K[i, j] * x[j] for j in range(n)) for i in range(m)]

    u_clamped = [
        d.if_then_else(u_lqr[i] > MAX_ACTION, MAX_ACTION,
            d.if_then_else(u_lqr[i] < -MAX_ACTION, -MAX_ACTION, u_lqr[i]))
        for i in range(m)
    ]

    fxu = dynamics_fn_dreal(x, u_clamped)

    lie_derivative = sum(fxu[i] * V.Differentiate(x[i]) for i in range(n))

    x_norm_sq = sum(xi**2 for xi in x)

    violation_condition = d.And(
        V <= level,
        x_norm_sq >= eps**2,
        in_box(x, lb, ub, scale),
        lie_derivative >= 0
    )
    r1 = d.CheckSatisfiability(violation_condition, delta)

    boundary_condition = d.And(
        on_boundary(x, lb, ub, scale),
        V <= level
    )
    r2 = d.CheckSatisfiability(boundary_condition, delta)

    return r1, r2


def lqr_check_continuous(level, scale=2., eps=0.5, delta=1e-3):
    n = agent_lqr.P_np.shape[0]
    x = dreal_var(n)

    P = agent_lqr.P_np
    V = sum(x[i] * sum(P[i, j] * x[j] for j in range(n)) for i in range(n))

    K = agent_lqr.K_np
    m = K.shape[0]
    u_lqr = [-sum(K[i, j] * x[j] for j in range(n)) for i in range(m)]
    
    u_clamped = [
        d.if_then_else(u_lqr[i] > MAX_ACTION, MAX_ACTION, 
            d.if_then_else(u_lqr[i] < -MAX_ACTION, -MAX_ACTION, u_lqr[i]))
        for i in range(m)
    ]

    fxu = dynamics_fn_dreal(x, u_clamped)

    lie_derivative = sum(fxu[i] * V.Differentiate(x[i]) for i in range(n))

    x_norm_sq = sum(xi**2 for xi in x)

    violation_condition = d.And(
        V <= level,
        x_norm_sq >= eps**2,
        in_box(x, lb, ub, scale),
        lie_derivative >= 0
    )
    r1 = d.CheckSatisfiability(violation_condition, delta)

    boundary_condition = d.And(
        on_boundary(x, lb, ub, scale),
        V <= level
    )
    r2 = d.CheckSatisfiability(boundary_condition, delta)

    return r1, r2

def lqr_check(level, scale=2., eps=0.5, delta=1e-4):
    if agent_lqr.discrete_discounted is True:
        return lqr_check_discrete(level, scale, eps, delta)
    else:
        return lqr_check_continuous(level, scale, eps, delta)


c_star = bisection(lqr_check, 2, eps=0.2)
print(f"\nLQR certified c* = {c_star:.4f}")
