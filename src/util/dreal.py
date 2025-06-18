import dreal as d
import numpy as np

def dreal_var(n: int, prefix="x"):
    return np.array([d.Variable(f"{prefix}{i}") for i in range(n)])

def dreal_elementwise(x, func):
    return np.array([func(xi) for xi in x])

def dreal_sigmoid(z):
    return 1 / (1 + d.exp(-z))

def in_box(x, lb, ub, scale=1.0):
    return d.And(*[d.And(x[i] >= lb[i]*scale, x[i] <= ub[i]*scale)
                   for i in range(len(x))])

def on_boundary(x, lb, ub, scale=2.0, pad=0.01):
    outer = in_box(x, lb, ub, scale)
    inner = in_box(x, lb, ub, scale*(1-pad))
    return d.And(outer, d.Not(inner))

def on_boundry_dreal(self, x, scale=2.0):
    condition1 = d.And(
        x[0] >= self.lb[0] * scale * 0.99,
        x[0] <= self.ub[0] * scale * 0.99,
        x[1] >= self.lb[1] * scale * 0.99,
        x[1] <= self.ub[1] * scale * 0.99
    )
    condition2 = d.Not(
        d.And(
            x[0] >= self.lb[0] * scale * 0.97,
            x[0] <= self.ub[0] * scale * 0.97,
            x[1] >= self.lb[1] * scale * 0.97,
            x[1] <= self.ub[1] * scale * 0.97
        )
    )
    return d.And( condition1, condition2 )

def is_unsat(result) -> bool:
    """True iff dReal returned UNSAT."""
    return result is None or (isinstance(result, str) and result.lower() == "unsat")


def dreal_in_circle(x: list, radius: float) -> d.Formula:
    x_norm_sq = sum(xi**2 for xi in x)
    return x_norm_sq <= radius**2


def dreal_on_circle_boundary(x: list, radius: float) -> d.Formula:
    """
    Returns a dreal formula for checking if a point x is on the boundary of a circle.
    NOTE: Using a small tolerance band can be more robust for SMT solvers.
    """
    x_norm_sq = sum(xi**2 for xi in x)
    # A strict equality can be hard for solvers. A thin band is often better.
    tolerance = 1e-4
    return d.And(
        x_norm_sq <= (radius + tolerance)**2,
        x_norm_sq >= (radius - tolerance)**2
    )


def extract_ce_from_model(model, state_dim):
    if not model:
        return np.zeros(state_dim)

    ce = np.zeros(state_dim)
    
    for var, val in model.items():
        var_name = str(var)
        
        if var_name.startswith('x'):
            try:
                i = int(var_name[1:])
                
                if 0 <= i < state_dim:
                    if isinstance(val, d.Interval):
                        ce[i] = val.mid()
                    else:
                        ce[i] = val
            except (ValueError, IndexError):
                continue
                
    return ce