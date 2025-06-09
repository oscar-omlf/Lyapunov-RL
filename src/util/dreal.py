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
