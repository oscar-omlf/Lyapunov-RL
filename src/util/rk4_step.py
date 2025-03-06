
def rk4_step(f, x, u, dt):
    """RK4 integration step for the continuous-time dynamics."""
    f1 = f(x, u)
    f2 = f(x + 0.5 * dt * f1, u)
    f3 = f(x + 0.5 * dt * f2, u)
    f4 = f(x + dt * f3, u)
    return x + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)
