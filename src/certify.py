import numpy as np
import dreal as d


from util.dreal import dreal_var, in_box, on_boundary, is_unsat
from agents.agent_factory import AgentFactory 
from util.dynamics import pendulum_dynamics_torch, pendulum_dynamics_dreal, double_integrator_dynamics_dreal, vanderpol_dynamics_torch, vanderpol_dynamics_dreal


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


config_lac_pendulum = {
    "agent_str": "Lyapunov-AC",
    "model_name": "LAC",
    "alpha": 0.2,
    "lr": 2e-3,
    "dynamics_fn": pendulum_dynamics_torch,
    "dynamics_fn_dreal": pendulum_dynamics_dreal,
    "batch_size": 128,
    "num_paths_sampled": 8,
    "dt": 0.003,
    "norm_threshold": 5e-2,
    "integ_threshold": 500,
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "actor_hidden_sizes": (5, 5),
    "critic_hidden_sizes": (20, 20),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "max_action": 1.0
}

config_lac_vanderpol = {
    "agent_str": "Lyapunov-AC",
    "model_name": "LAC_CEGAR",
    "environment": "VanDerPol",
    "alpha": 0.1,
    "lr": 3e-3,
    "dynamics_fn": vanderpol_dynamics_torch,
    "dynamics_fn_dreal": vanderpol_dynamics_dreal,
    "batch_size": 128,
    "num_paths_sampled": 8,
    "dt": 0.01,
    "norm_threshold": 5e-2,
    "integ_threshold": 150,
    "r1_bounds": (np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
    "actor_hidden_sizes": (30, 30), 
    "critic_hidden_sizes": (30, 30),
    "state_space": np.zeros(2), 
    "action_space": np.zeros(1), 
    "max_action": 1.0,
    "normalize_gradients": True
}

config_lqr_pendulum = {
    "agent_str": "LQR",
    "environment": "InvertedPendulum",
    "discrete_discounted": True,
    "gamma": 0.99,
    "dt": 0.003,
    "g": 9.81,
    "m": 0.15,
    "l": 0.5,
    "b": 0.1,
    "max_action": 1.0,
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
}

config_lqr_vanderpol = {
    "agent_str": "LQR",
    "environment": "VanDerPol",
    "discrete_discounted": False,
    "gamma": 0.99,
    "dt": 0.01,
    "mu": 1.0,
    "max_action": 1.0,
    "R": 0.1 * np.eye(1),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
}

CFG_LAC = config_lac_pendulum
CFG_LQR = config_lqr_pendulum

# agent_lac = AgentFactory.create_agent(config=config_lac_pendulum)
agent_lqr = AgentFactory.create_agent(config=CFG_LQR)

# agent_lac.load(file_path='best_models/', episode=0.94265625)

# agent_lac.trainer._sanity_network()
# agent_lac.trainer._sanity_dynamics()

# r = agent_lac.trainer.check_lyapunov_with_ce(level=0.5, scale=2.0, eps=0.5)
# print(r)

# c_star = bisection(agent_lac.trainer.check_lyapunov, LEVEL_INIT)
# print(f"\nLyAC certified c* = {c_star:.4f}")

alpha = CFG_LAC["alpha"]
P     = agent_lqr.P_np
R1_LB, R1_UB = CFG_LAC["r1_bounds"]
lb    = R1_LB
ub    = R1_UB

print(f"Alpha: {alpha}, R1_LB: {lb}, R1_UB: {ub}")  

def lqr_check_discrete(level, scale=2., eps=0.5, delta=1e-4):
    n = agent_lqr.P_np.shape[0]
    x = dreal_var(n)

    P = agent_lqr.P_np
    V = sum(x[i] * sum(P[i, j] * x[j] for j in range(n)) for i in range(n))

    K = agent_lqr.K_np
    u = [-sum(K[i, j] * x[j] for j in range(n)) for i in range(K.shape[0])]

    f_xu = pendulum_dynamics_dreal(x, u)
    x_next = [x[i] + agent_lqr.dt * f_xu[i] for i in range(n)]
    # x_next = pendulum_dynamics_dreal(x, u)
    V_next = sum(x_next[i] * sum(P[i, j] * x_next[j] for j in range(n)) for i in range(n))

    # lyapunov_violation = (agent_lqr.gamma * V_next - V >= 0)
    Qg = agent_lqr.Q_gamma
    Rg = agent_lqr.R_gamma
    m = len(u)

    quad_cost = sum(x[i] * sum(Qg[i,j] * x[j] for j in range(n)) for i in range(n)) \
          + sum(u[i] * sum(Rg[i,j] * u[j] for j in range(m)) for i in range(m))


    lhs = agent_lqr.gamma * V_next - V + quad_cost
    lyapunov_violation = lhs >= 1e-6
    
    x_norm_sq = sum(xi**2 for xi in x)

    r1 = d.CheckSatisfiability(
        d.And(
            V <= level,
            x_norm_sq >= eps**2,
            in_box(x, lb, ub, scale),
            lyapunov_violation
        ),
        delta)

    r2 = d.CheckSatisfiability(
        d.And(
            on_boundary(x, lb, ub, scale),
            V <= level
        ),
        delta)

    return r1, r2


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

    fxu = pendulum_dynamics_dreal(x, u_clamped)

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

    # x_next = A_d x + B_d u  (A_d,B_d are numeric constants)
    A = agent_lqr.A_np
    B = agent_lqr.B_np
    x_next = [ sum(A[i, j] * x[j] for j in range(n)) +
               sum(B[i, k] * u[k] for k in range(m))
               for i in range(n) ]

    V_next = sum(x_next[i] * sum(P[i, j] * x_next[j]
                                 for j in range(n))
                 for i in range(n))

    Qg, Rg = agent_lqr.Q_gamma, agent_lqr.R_gamma
    stage = (sum(x[i] * sum(Qg[i, j] * x[j] for j in range(n))
                 for i in range(n)) +
             sum(u[i] * sum(Rg[i, j] * u[j] for j in range(m))
                 for i in range(m)))

    lhs = agent_lqr.gamma * V_next - V + stage
    lyap_violation = lhs >= 1e-5

    x_norm_sq = sum(xi**2 for xi in x)

    r1 = d.CheckSatisfiability(
        d.And(V <= level,
              x_norm_sq >= eps**2,
              in_box(x, lb, ub, scale),
              lyap_violation),
        delta)

    r2 = d.CheckSatisfiability(
        d.And(on_boundary(x, lb, ub, scale),
              V <= level),
        delta)

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

    fxu = pendulum_dynamics_dreal(x, u_clamped)

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
