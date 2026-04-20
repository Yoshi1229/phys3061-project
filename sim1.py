import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# Constants
GMsun = 4 * np.pi**2          # AU^3 / yr^2

muJ   = 0.0009543 * GMsun     # Jupiter mass ratio (more accurate than 0.001)
aJ    = 5.2                   # Jupiter semi-major axis (AU)
eJ    = 0.0484                # 0.0484 for real Jupiter
nJ    = np.sqrt(GMsun / aJ**3)
TJ    = 2 * np.pi / nJ        # ~11.86 years

aS    = 9.58                  # Saturn semi-major axis (AU)
muS   = 0.0002856 * GMsun     # Saturn mass ratio (Change to 0.0 to run without Saturn)
eS    = 0.0565                # Saturn eccentricity
nS    = np.sqrt(GMsun / aS**3)
TS    = 2 * np.pi / nS        # ~29.46 years

# Simulation parameters
N = 10000                     # number of asteroids
t_end = 2500000.0             # time frame (yr)
dt = 0.25                     # time step (yr)
sample_dt = 5000.0            # sample rate (yr)

# Jupiter orbit (2D planar, Keplerian)
@njit
def jupiter_state(t):
    M = nJ * t # Mean anomaly

    # Solve for Eccentric Anomaly (E) using Newton-Raphson method
    E = M
    for _ in range(5):
        E = E - (E - eJ * np.sin(E) - M) / (1.0 - eJ * np.cos(E))

    # Position (heliocentric)
    cosE = np.cos(E); sinE = np.sin(E)
    fac = np.sqrt(1.0 - eJ*eJ)
    x = aJ * (cosE - eJ)
    y = aJ * fac * sinE
    RJ = np.array([x, y])

    # Velocity (heliocentric)
    Edot = nJ / (1.0 - eJ * np.cos(E))
    vx = -aJ * sinE * Edot
    vy =  aJ * fac * cosE * Edot
    VJ = np.array([vx, vy])
    return RJ, VJ

# Saturn orbit
@njit
def saturn_state(t):
    M = nS * t
    E = M
    for _ in range(5):
        E = E - (E - eS * np.sin(E) - M) / (1.0 - eS * np.cos(E))
    cosE = np.cos(E); sinE = np.sin(E)
    fac = np.sqrt(1.0 - eS*eS)
    x = aS * (cosE - eS)
    y = aS * fac * sinE
    RS = np.array([x, y])
    Edot = nS / (1.0 - eS * np.cos(E))
    vx = -aS * sinE * Edot
    vy =  aS * fac * cosE * Edot
    VS = np.array([vx, vy])
    return RS, VS

# Equations of motion (massless asteroids)
@njit
def accel_restricted(r, t):
    RJ, _ = jupiter_state(t)
    RS, _ = saturn_state(t)

    a = np.zeros_like(r)

    RJ2 = RJ[0]*RJ[0] + RJ[1]*RJ[1]; RJ3 = RJ2**1.5 + 1e-30
    RS2 = RS[0]*RS[0] + RS[1]*RS[1]; RS3 = RS2**1.5 + 1e-30

    a_ind_x = -muJ * RJ[0] / RJ3 - muS * RS[0] / RS3
    a_ind_y = -muJ * RJ[1] / RJ3 - muS * RS[1] / RS3

    for i in range(r.shape[0]):
        rx = r[i,0]; ry = r[i,1]

        r2 = rx*rx + ry*ry
        r3 = r2**1.5 + 1e-30

        # Jupiter
        dxJ = rx - RJ[0]; dyJ = ry - RJ[1]
        drJ2 = dxJ*dxJ + dyJ*dyJ
        drJ3 = drJ2**1.5 + 1e-30

        # Saturn
        dxS = rx - RS[0]; dyS = ry - RS[1]
        drS2 = dxS*dxS + dyS*dyS
        drS3 = drS2**1.5 + 1e-30

        ax = -GMsun * rx / r3 - muJ * dxJ / drJ3 - muS * dxS / drS3 + a_ind_x
        ay = -GMsun * ry / r3 - muJ * dyJ / drJ3 - muS * dyS / drS3 + a_ind_y

        a[i,0] = ax
        a[i,1] = ay
    return a

# util functions
def semi_major_axis(r, v):
    # estimate a by 2-body orbital energy (asteroid and sun)
    # a = (GM) / [(2GM/r)-v^2] 
    rnorm = np.linalg.norm(r, axis=-1)
    v2 = np.sum(v*v, axis=-1)
    denom = (2*GMsun/rnorm - v2)
    a = np.where(denom > 0, GMsun / denom, np.nan)
    return a

def osculating_eccentricity(r, v, a):
    # estimate eccentricity for 2D planar orbit
    hz = r[:,0]*v[:,1] - r[:,1]*v[:,0]
    e_sq = np.maximum(1.0 - (hz*hz) / (GMsun * np.maximum(a, 1e-300)), 0.0)
    return np.sqrt(e_sq)

rng = np.random.default_rng(42)

def init_asteroids(N=10000, a_min=2.00, a_max=4.00):
    a0 = rng.uniform(a_min, a_max, N) # place asteroids between min and max semi-major axis
    theta0 = rng.uniform(0, 2*np.pi, N) # place them randomly along their orbits

    r0 = np.stack([a0*np.cos(theta0), a0*np.sin(theta0)], axis=1) # position
    v_circ = np.sqrt(GMsun / a0)
    v0 = np.stack([-v_circ*np.sin(theta0), v_circ*np.cos(theta0)], axis=1) # velocity

    return r0, v0, a0

# integrate equations of motion and sample snapshots
def integrate_and_sample(r0, v0, t_end=2500000.0, dt=0.25, sample_dt=5000.0,
                         r_min=1.5, r_max=50.0, target_idx=None):
    r = r0.copy()
    v = v0.copy()
    t = 0.0
    a_now = accel_restricted(r, t)

    # sampling bookkeeping
    next_sample = sample_dt
    times = []
    a_samples = []
    qmin = np.full(r.shape[0], np.inf)
    keep_mask = np.ones(r.shape[0], dtype=np.bool_)

    # target asteroid time-series (for asteroid plot)
    t_series = [] # time
    a_series = [] # semi-major axis
    e_series = [] # eccentricity
    r_series = [] # distance
    q_series = [] # perihelion

    steps = int(np.ceil(t_end/dt))
    for n in range(steps):
        # Velocity-Verlet algorithm
        r += v*dt + 0.5*a_now*(dt*dt)
        t += dt
        a_new = accel_restricted(r, t)
        v += 0.5*(a_now + a_new)*dt
        a_now = a_new

        # survival mask
        rn = np.hypot(r[:,0], r[:,1])
        keep_mask &= (rn < r_max) & (rn > r_min) # eliminate asteroids that are too close (collision) or too far (ejection)

        # sample
        if t >= next_sample or n == steps-1:
            times.append(t)
            a_os = semi_major_axis(r, v)
            e_os = osculating_eccentricity(r, v, a_os)
            q_os = a_os * (1.0 - e_os)
            
            good = np.isfinite(q_os) # only keep non-ejected asteroids
            qmin[good] = np.minimum(qmin[good], q_os[good])

            # store only a(t) per sample to compute medians later
            a_snap = a_os.copy()
            a_snap[~keep_mask] = np.nan
            a_samples.append(a_snap)

            # record target's series to plot
            if target_idx is not None:
                t_series.append(t)
                a_t = a_os[target_idx]
                e_t = e_os[target_idx]
                r_t = rn[target_idx]
                q_t = q_os[target_idx]
                a_series.append(a_t)
                e_series.append(e_t)
                r_series.append(r_t)
                q_series.append(q_t)

            next_sample += sample_dt

    A = np.stack(a_samples, axis=0)  # (K,N)
    return (np.array(times), A, qmin, keep_mask,
            np.array(t_series), np.array(a_series), np.array(e_series),
            np.array(r_series), np.array(q_series))


# Simulation run begins here
# init
r0, v0, a0_init = init_asteroids(N=N, a_min=2.00, a_max=4.00)

# pick asteroid closest to the 3:1 resonance
a_31 = aJ * (1.0/3.0)**(2.0/3.0)
target_idx = int(np.argmin(np.abs(a0_init - a_31)))

# integrate and sample
(times, A, qmin, keep_mask,
 t_series, a_series, e_series, r_series, q_series) = integrate_and_sample(
    r0, v0, t_end=t_end, dt=dt, sample_dt=sample_dt,
    r_min=1.5, r_max=50.0, target_idx=target_idx
)

# Plotting
# single asteroid plot (near 3:1)
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, dpi=150)

axs[0].plot(t_series, a_series, color='steelblue', lw=1.5)
axs[0].set_ylabel('Semi-major Axis (a) (AU)', fontsize=11)
axs[0].set_title('Orbital Evolution of an Asteroid near 3:1', fontsize=14, fontweight='bold')
axs[0].grid(True, alpha=0.3)

axs[1].plot(t_series, e_series, color='firebrick', lw=1.5)
axs[1].set_ylabel('Eccentricity (e)', fontsize=11)
axs[1].grid(True, alpha=0.3)

axs[2].plot(t_series, r_series, color='forestgreen', alpha=0.3, lw=1, label='Instantaneous Distance (r)')
axs[2].plot(t_series, q_series, color='darkgreen', lw=2, label='Perihelion (q)')
axs[2].axhline(1.5, color='black', linestyle='--', lw=1.5, label='Mars Orbit (1.5 AU)')
axs[2].set_ylabel('Distance (AU)', fontsize=11)
axs[2].set_xlabel('Time (years)', fontsize=11)
axs[2].legend(loc='upper right')
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Resonant_Asteroid.png', bbox_inches='tight')
plt.show()

# histogram
K = A.shape[0]
A_last = A[2*K//3:] # last third of samples
a_median = np.nanmedian(A_last, axis=0)

# additional filter to keep reasonably bounded survivors; apply Mars-crossing cut
survivors = np.isfinite(a_median) & keep_mask & (a_median < 4.5) #& (qmin > 1.5)
a_plot = a_median[survivors]

def resonance_a(p, q, aJ=aJ):
    return aJ * (q/float(p))**(2.0/3.0)

res_list = [(3,1), (5,2), (7,3), (2,1),
            (5,3), (7,4)]   # include 5:3 (~3.70 AU) and 7:4 (~3.58 AU)

plt.figure(figsize=(10, 5), dpi=120)
bins = np.linspace(2.0, 4.0, 90)
plt.hist(a_plot, bins=bins, color='steelblue', edgecolor='black', linewidth=0.3, alpha=0.85)

for (p,q) in res_list:
    ax = resonance_a(p,q)
    if 2.0 <= ax <= 4.0:
        plt.axvline(ax, color='k', ls='--', lw=1.0)
        plt.text(ax+0.012, plt.ylim()[1]*0.85, f'{p}:{q}',
                 rotation=90, va='top', ha='left', fontsize=9, fontweight='bold')

plt.xlabel('Semi-major axis a (AU)', fontsize=11)
plt.ylabel('Number of surviving asteroids', fontsize=11)
plt.title('Asteroid distribution and Kirkwood gaps (planar, Jupiter + Saturn)', fontsize=13)
plt.xlim(2.0, 4.0)
plt.tight_layout()
plt.show()
