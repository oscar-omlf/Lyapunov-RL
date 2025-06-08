import numpy as np
import dreal as d


from util.dreal import dreal_var, in_box, on_boundary, is_unsat
from agents.agent_factory import AgentFactory 
from util.dynamics import pendulum_dynamics_torch, pendulum_dynamics_dreal, double_integrator_dynamics_dreal, vanderpol_dynamics_torch, vanderpol_dynamics_dreal


LEVEL_INIT = 0.95

def bisection(check_fun, c_max=0.95, tol=1e-3, it_max=40):
    hi_fail = c_max
    while True:
        r1, r2 = check_fun(hi_fail, eps=0.5)
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
        r1, r2 = check_fun(mid, eps=0.5)
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
    "batch_size": 64,
    "num_paths_sampled": 8,
    "dt": 0.003,
    "norm_threshold": 5e-2,
    "integ_threshold": 150,
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "actor_hidden_sizes": (5, 5),
    "critic_hidden_sizes": (20, 20),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "max_action": 1.0
}

config_lqr = {
    "agent_str": "LQR",
    'environment': 'InvertedPendulum',
    'discrete':    False,
    'g':           9.81,
    'm':           0.15,
    'l':           0.5,
    'max_action':  1.0,
    'state_space': np.zeros(2),
    'action_space':np.zeros(1),
}


agent_lac = AgentFactory.create_agent(config=config_lac_pendulum)
agent_lqr = AgentFactory.create_agent(config=config_lqr)

agent_lac.load(file_path='./logs/LAC/run_1/', episode=3000)

c_star = bisection(agent_lac.trainer.check_lyapunov, LEVEL_INIT)
print(f"\nLyAC certified c* = {c_star:.4f}")

alpha = 0.2
P     = agent_lqr.P_np
lb    = np.array([-2., -4.])
ub    = np.array([ 2.,  4.])

def lqr_check(level, scale=2., eps=0.5, delta=1e-4):
    x = dreal_var(2)
    V = sum(x[i]*sum(P[i,j]*x[j] for j in range(2)) for i in range(2))
    W = d.tanh(alpha * V)
    u = - (agent_lqr.K_np @ np.array(x))[0]
    fx = pendulum_dynamics_dreal(x, np.array([u]))
    lie = sum(fx[i] * W.Differentiate(x[i]) for i in range(2))
    xnorm = d.sqrt(x[0]*x[0]+x[1]*x[1])

    W0 = 0.0

    r1 = d.CheckSatisfiability(
            d.And(xnorm>=eps, in_box(x,lb,ub,scale),
                    W<=level, d.Or(lie>=0, W<=W0)),
            delta)
    r2 = d.CheckSatisfiability(
            d.And(on_boundary(x,lb,ub,scale), W<=level), delta)
    return r1, r2

c_star = bisection(lqr_check, LEVEL_INIT)
print(f"\nLQR certified c* = {c_star:.4f}")
