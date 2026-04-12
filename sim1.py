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

def integrate_restricted(r0, v0, t_end=200*np.pi, dt=0.002, sample_every=200):
    """
    t_end default ~ 200*pi yrs (~100 Jupiter orbits since TJ=2*pi)
    """
    r = r0.copy()
    v = v0.copy()
    t = 0.0
    a = accel_restricted(r, t)
    snapshots = []   # store (t, r, v) sparsely
    keep_mask = np.ones(len(r), dtype=bool)

    # Ejection criteria parameters
    r_max = 50.0  # AU; beyond this treat as ejected
    r_min = 0.2   # AU; collisions with Sun

    steps = int(np.ceil(t_end/dt))
    for n in range(steps):
        # r_{n+1}
        r += v*dt + 0.5*a*(dt**2)
        t += dt
        # a_{n+1}
        a_new = accel_restricted(r, t)
        # v_{n+1}
        v += 0.5*(a + a_new)*dt
        a = a_new

        # Ejection/collision masking (optional)
        rn = np.linalg.norm(r, axis=1)
        keep_mask &= (rn < r_max) & (rn > r_min)

        if (n % sample_every) == 0 or n == steps-1:
            snapshots.append((t, r.copy(), v.copy(), keep_mask.copy()))

    return snapshots

