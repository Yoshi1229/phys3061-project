import numpy as np
import matplotlib.pyplot as plt

# Units: AU, years, Msun
GMsun = 4 * np.pi**2       # AU^3 / yr^2
muJ   = 0.001 * GMsun      # GM of Jupiter (mass ratio ~1e-3)
rJ    = 5.2                # AU
TJ    = 2*np.pi            # yrs
nJ    = 2*np.pi / TJ       # = 1 rad/yr

def jupiter_state(t):
    # Prescribed circular orbit in the restricted model
    c, s = np.cos(nJ*t), np.sin(nJ*t)
    RJ = np.array([rJ*c, rJ*s])                       # position
    VJ = nJ * np.array([-rJ*s, rJ*c])                 # velocity
    return RJ, VJ

def accel_restricted(r_ast, t):
    # r_ast: (..., 2) array of asteroid positions
    RJ, _ = jupiter_state(t)
    # Sun’s acceleration
    r = r_ast
    r2 = np.sum(r*r, axis=-1, keepdims=True)
    a_sun = -GMsun * r / (r2**1.5 + 1e-30)
    # Jupiter’s acceleration
    dr = r - RJ
    dr2 = np.sum(dr*dr, axis=-1, keepdims=True)
    a_jup = -muJ * dr / (dr2**1.5 + 1e-30)
    return a_sun + a_jup

def semi_major_axis(r, v):
    # Using a = GM / (2GM/r - v^2)
    rnorm = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v*v, axis=-1)
    denom = (2*GMsun/rnorm - v2)
    # Guard against division by zero or unbound orbits (denom<=0)
    a = np.where(denom > 0, GMsun / denom, np.nan)
    return a

rng = np.random.default_rng(1)

def init_asteroids(N=2000, a_min=2.0, a_max=4.0, phase='aligned'):
    if phase == 'aligned':
        a0 = np.linspace(a_min, a_max, N)
        theta0 = np.zeros(N)
    else:
        a0 = rng.uniform(a_min, a_max, N)
        theta0 = rng.uniform(0, 2*np.pi, N)
    r0 = np.stack([a0*np.cos(theta0), a0*np.sin(theta0)], axis=1)
    v_circ = np.sqrt(GMsun / a0)
    # velocity perpendicular to r pointing +phi
    v0 = np.stack([-v_circ*np.sin(theta0), v_circ*np.cos(theta0)], axis=1)
    return r0, v0, a0

