import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.patheffects as pe


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 50,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _time_to_steps(solver, times_years):
    dt = solver.params.dt
    return [int(round(t / dt)) for t in times_years]


def _pick_c_index(solver, c_fixed=None):
    if c_fixed is None:
        c_fixed = solver.c_grid[len(solver.c_grid) // 2]
    idx_c = np.argmin(np.abs(solver.c_grid - c_fixed))
    return idx_c, solver.c_grid[idx_c]


def plot_value_slices(solver, c_fixed=None, times_years=(0.0, 2.0, 5.0, 9.0)):
    if not hasattr(solver, "value"):
        raise RuntimeError("Run solver.solve() first.")

    idx_c, c_used = _pick_c_index(solver, c_fixed)
    time_steps = _time_to_steps(solver, times_years)
    time_steps = [t for t in time_steps if 0 <= t < solver.value.shape[0]]

    if len(time_steps) == 0:
        raise ValueError("No valid time steps to plot.")

    # we want exactly 4 panels if possible
    if len(time_steps) > 4:
        time_steps = time_steps[:4]

    vals = [solver.value[t, :, idx_c, :] for t in time_steps]
    vmin = min(v.min() for v in vals)
    vmax = max(v.max() for v in vals)

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10),
        constrained_layout=True
    )
    axes = axes.ravel()

    extent = [
        solver.r_grid[0], solver.r_grid[-1],
        solver.k_grid[0], solver.k_grid[-1]
    ]

    im = None
    for ax, t, V in zip(axes, time_steps, vals):
        im = ax.imshow(
            V.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        ax.axhline(
            solver.params.c0,
            linestyle="--",
            color="white",
            linewidth=2,
            label="default threshold",
        )

        ax.axhline(
            solver.params.c_star,
            linestyle=":",
            color="red",
            linewidth=2,
            label="target capital",
        )
        ax.set_title(f"t = {t * solver.params.dt:.1f} years")
        ax.set_xlabel("short rate r")
        ax.set_ylabel("capital ratio k")
        ax.legend(loc="upper left")
        ax.grid(False)

    # hide unused panels if less than 4 time steps
    for ax in axes[len(time_steps):]:
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes[:len(time_steps)], shrink=0.95, pad=0.02)
    cbar.set_label("value function")

    fig.suptitle(f"Value function slices, fixed coupon c = {c_used:.3f}", y=1.02)
    plt.show()


def plot_policy_maps(solver, t=0, c_fixed=None):

    if not hasattr(solver, "opt_g"):
        raise RuntimeError("Run solver.solve() first.")

    idx_c, c_used = _pick_c_index(solver, c_fixed)

    g = solver.opt_g[t, :, idx_c, :]
    p = solver.opt_p[t, :, idx_c, :]

    extent = [solver.r_grid[0], solver.r_grid[-1], solver.k_grid[0], solver.k_grid[-1]]

    fig, axes = plt.subplots(2, 1, figsize=(9, 10), constrained_layout=True)

    im1 = axes[0].imshow(
        g.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    # axes[0].axhline(solver.params.c0, linestyle="--", color="white", linewidth=1.8)
    # axes[0].axhline(solver.params.c_star, linestyle=":", color="red", linewidth=1.8)
    axes[0].set_title(f"Optimal loan growth g* at t = {t * solver.params.dt:.1f} years, c = {c_used:.3f}")
    axes[0].set_xlabel("short rate r")
    axes[0].set_ylabel("capital ratio k")
    axes[0].grid(False)

    cbar1 = fig.colorbar(im1, ax=axes[0], shrink=0.92, pad=0.02)
    cbar1.set_label("g*")

    # make p-map visually less harsh for all-zero regions
    im2 = axes[1].imshow(
        p.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
    )
    for ax in axes:
        ax.axhline(
            solver.params.c0,
            linestyle="--",
            color="black",
            linewidth=2,
            label="default threshold",
        )

        ax.axhline(
            solver.params.c_star,
            linestyle=":",
            color="red",
            linewidth=2,
            label="target capital",
        )

        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="black",
        )

    axes[1].set_title(f"Optimal payout ratio p* at t = {t * solver.params.dt:.1f} years, c = {c_used:.3f}")
    axes[1].set_xlabel("short rate r")
    axes[1].set_ylabel("capital ratio k")
    axes[1].grid(False)

    cbar2 = fig.colorbar(im2, ax=axes[1], shrink=0.92, pad=0.02)
    cbar2.set_label("p*")

    leg = axes[1].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="black")
    for text in leg.get_texts():
        text.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

    # fig.suptitle(f"Policy maps, fixed coupon c = {c_used:.3f}", y=1.01)
    plt.show()


def plot_policy_slices(solver, r_fixed=0.05, c_fixed=0.08, t=0):

    if not hasattr(solver, "opt_g"):
        raise RuntimeError("Run solver.solve() first.")

    idx_r = np.argmin(np.abs(solver.r_grid - r_fixed))
    idx_c = np.argmin(np.abs(solver.c_grid - c_fixed))

    r_used = solver.r_grid[idx_r]
    c_used = solver.c_grid[idx_c]

    g = solver.opt_g[t, idx_r, idx_c, :]
    p = solver.opt_p[t, idx_r, idx_c, :]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    axes[0].plot(solver.k_grid, g, linewidth=2.5, color="tab:blue", label=r"$g^*(k)$")
    axes[0].axhline(solver.params.alpha_annual, linestyle="--", linewidth=2, color="tab:orange", label=r"$\alpha_{\mathrm{step}}$")
    axes[0].axvline(solver.params.c0, linestyle=":", linewidth=2, color="tab:red", label=r"$c_0$")
    axes[0].axvline(solver.params.c_star, linestyle="-.", linewidth=2, color="tab:green", label=r"$c^*$")
    axes[0].set_title(f"Growth policy at r={r_used:.3f}, c={c_used:.3f}, t={t * solver.params.dt:.1f}y")
    axes[0].set_xlabel("capital ratio k")
    axes[0].set_ylabel("loan growth g (per step)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(solver.k_grid, p, linewidth=2.5, color="tab:purple", label=r"$p^*(k)$")
    axes[1].axvline(solver.params.c0, linestyle=":", linewidth=2, color="tab:red", label=r"$c_0$")
    axes[1].axvline(solver.params.c_star, linestyle="-.", linewidth=2, color="tab:green", label=r"$c^*$")
    axes[1].set_title(f"Payout policy at r={r_used:.3f}, c={c_used:.3f}, t={t * solver.params.dt:.1f}y")
    axes[1].set_xlabel("capital ratio k")
    axes[1].set_ylabel("payout ratio p")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.show()


def plot_spread_curve(solver):

    g = np.asarray(solver.params.g_actions)

    spread = solver.params.spread(g)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))

    ax.plot(
        g * 100,
        spread * 100,
        linewidth=2.5,
        label="spread curve",
    )

    ax.axhline(
        solver.params.s0_annual * 100,
        linestyle=":",
        linewidth=2,
        color="tab:orange",
        label=r"$s_0$",
    )

    ax.axhline(
        solver.params.s_max_annual * 100,
        linestyle="--",
        linewidth=2,
        color="tab:red",
        label=r"$s_{\max}$",
    )

    ax.axvline(
        solver.params.alpha_annual * 100,
        linestyle="-.",
        linewidth=2,
        color="tab:green",
        label=r"$\alpha$",
    )

    ax.set_xlabel("loan growth (% p.a.)")
    ax.set_ylabel("spread (% p.a.)")
    ax.set_title("Spread on new loans as a function of growth")

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.show()


def _plot_quantile_band(ax, time, data, title, ylabel):
    q10 = np.quantile(data, 0.10, axis=1)
    q25 = np.quantile(data, 0.25, axis=1)
    q50 = np.quantile(data, 0.50, axis=1)
    q75 = np.quantile(data, 0.75, axis=1)
    q90 = np.quantile(data, 0.90, axis=1)

    ax.fill_between(time, q10, q90, alpha=0.18, label="10–90%")
    ax.fill_between(time, q25, q75, alpha=0.35, label="25–75%")
    ax.plot(time, q50, linewidth=2, label="median")
    ax.set_title(title)
    ax.set_xlabel("time (years)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_simulation_quantiles(sim):
    """
    Clean 2x2 panel with median and quantile bands.
    """

    time = sim["time_grid"]
    rates = sim["rates"]
    capital = sim["capital_ratio"]
    coupons = sim["coupons"]
    dividends = sim["dividends_norm"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    _plot_quantile_band(axes[0, 0], time, rates, "Short rate", "r")
    _plot_quantile_band(axes[0, 1], time, capital, "Capital ratio", "k")
    _plot_quantile_band(axes[1, 0], time, coupons, "Average coupon", "c")
    _plot_quantile_band(axes[1, 1], time, dividends, "Normalized dividends", "d")

    fig.suptitle("Simulation summary under optimal policy", y=1.02)
    plt.show()


def plot_survival_curve(sim):
    """
    Survival probability over time.
    """

    default_times = sim["default_times"]
    time = sim["time_grid"]
    N = len(time) - 1

    survival = np.array([(default_times > t).mean() for t in range(N + 1)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, survival, linewidth=2, label="survival probability")
    ax.set_title("Survival curve")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("P(no default by t)")
    ax.set_ylim(0.0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_default_histogram(sim):
    """
    Histogram of default times in years.
    """

    default_times = sim["default_times"]
    dt = sim["time_grid"][1] - sim["time_grid"][0]
    N = len(sim["time_grid"]) - 1

    defaults = default_times[default_times <= N]
    if len(defaults) == 0:
        print("No defaults in the simulation sample.")
        return

    default_years = defaults * dt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(default_years, bins=20, edgecolor="black", alpha=0.85, label="default times")
    ax.set_title("Distribution of default times")
    ax.set_xlabel("time of default (years)")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def bellman_residual(solver, value, opt_g, opt_p, t, r, c, k):
    """
    Pointwise Bellman residual at a single point.
    """
    i_r = np.argmin(np.abs(solver.r_grid - r))
    i_c = np.argmin(np.abs(solver.c_grid - c))
    i_k = np.argmin(np.abs(solver.k_grid - k))

    g = opt_g[t, i_r, i_c, i_k]
    p = opt_p[t, i_r, i_c, i_k]

    cache = solver.action_cache[(g, p)]
    m = cache["m"]
    pi = cache["pi"][i_r, i_c, i_k]
    d = cache["d"][i_r, i_c, i_k]
    default_flag = cache["default"][i_r, i_c, i_k]
    k_next = cache["k_next"][i_r, i_c, i_k]
    c_next = cache["c_next"][i_r, i_c, i_k]

    cont = 0.0
    phi = solver.params.mean_reversion_factor
    mu = solver.params.mu_annual
    sd = solver.params.base_sd
    dt = solver.params.dt

    for z, w in zip(solver.z_nodes, solver.z_probs):
        r_prime = mu + phi * (r - mu) + sd * z
        r_prime = np.clip(r_prime, solver.params.r_min, solver.params.r_max)
        v_next = solver._interp3_uniform(
            value[t + 1], r_prime, c_next, k_next,
            solver.r_grid, solver.c_grid, solver.k_grid
        )
        cont += w * v_next

    if default_flag:
        rhs = (1.0 - solver.params.delta) * max(k + pi, 0.0)
    else:
        rhs = d + solver.params.gamma * m * cont

    lhs = value[t, i_r, i_c, i_k]
    return lhs - rhs


def plot_bellman_residual_heatmap(solver, t=0, c_fixed=None):
    """
    Heatmap of |Bellman residual| on an (r, k) slice.
    """

    if not hasattr(solver, "value"):
        raise RuntimeError("Run solver.solve() first.")

    idx_c, c_used = _pick_c_index(solver, c_fixed)

    residual = np.zeros((solver.nr, solver.nk), dtype=float)

    for i, r in enumerate(solver.r_grid):
        for j, k in enumerate(solver.k_grid):
            res = bellman_residual(
                solver,
                solver.value,
                solver.opt_g,
                solver.opt_p,
                t,
                r,
                c_used,
                k,
            )
            residual[i, j] = abs(res)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    im = ax.imshow(
        np.log10(residual.T + 1e-12),
        origin="lower",
        aspect="auto",
        extent=[solver.r_grid[0], solver.r_grid[-1], solver.k_grid[0], solver.k_grid[-1]],
        cmap="magma",
    )
    ax.set_title(f"Log10 Bellman residual, t={t * solver.params.dt:.1f}y, c={c_used:.3f}")
    ax.set_xlabel("short rate r")
    ax.set_ylabel("capital ratio k")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10 |residual|")
    plt.show()


def plot_bellman_residual_summary(solver, c_fixed=None, t_steps=(0, 24, 60, 96, 119), n_r=15, n_k=15):

    if not hasattr(solver, "value"):
        raise RuntimeError("Run solver.solve() first.")

    idx_c, c_used = _pick_c_index(solver, c_fixed)
    t_steps = [t for t in t_steps if 0 <= t < solver.value.shape[0]]

    max_res = []
    med_res = []

    r_idx = np.linspace(0, solver.nr - 1, n_r).astype(int)
    k_idx = np.linspace(0, solver.nk - 1, n_k).astype(int)

    for t in t_steps:
        residuals = []
        for i in r_idx:
            for j in k_idx:
                r = solver.r_grid[i]
                k = solver.k_grid[j]
                res = bellman_residual(
                    solver,
                    solver.value,
                    solver.opt_g,
                    solver.opt_p,
                    t,
                    r,
                    c_used,
                    k
                )
                residuals.append(abs(res))
        residuals = np.asarray(residuals)
        max_res.append(residuals.max())
        med_res.append(np.median(residuals))

    fig, ax = plt.subplots(figsize=(8.5, 5))
    times = np.array(t_steps) * solver.params.dt

    ax.plot(times, np.log10(np.array(max_res) + 1e-16), marker="o", linewidth=2, label="max log10|residual|")
    ax.plot(times, np.log10(np.array(med_res) + 1e-16), marker="s", linewidth=2, label="median log10|residual|")
    ax.set_title(f"Bellman residual summary, fixed c = {c_used:.3f}")
    ax.set_xlabel("time (years)")
    ax.set_ylabel("log10 residual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_bellman_residual_histogram(solver, c_fixed=None, t=0, n_samples=300):

    if not hasattr(solver, "value"):
        raise RuntimeError("Run solver.solve() first.")

    idx_c, c_used = _pick_c_index(solver, c_fixed)

    rng = np.random.default_rng(123)
    residuals = []

    for _ in range(n_samples):
        i = rng.integers(0, solver.nr)
        j = rng.integers(0, solver.nk)
        r = solver.r_grid[i]
        k = solver.k_grid[j]
        res = bellman_residual(
            solver,
            solver.value,
            solver.opt_g,
            solver.opt_p,
            t,
            r,
            c_used,
            k
        )
        residuals.append(abs(res))

    residuals = np.asarray(residuals)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.hist(np.log10(residuals + 1e-16), bins=25, edgecolor="black", alpha=0.85)
    ax.set_title(f"Bellman residual histogram, t={t * solver.params.dt:.1f}y, c={c_used:.3f}")
    ax.set_xlabel("log10 |residual|")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_loan_rate_distribution(solver, sim, times_years=(1.0, 3.0, 5.0, 9.0)):
    """
    Left panel: quantile bands of new‑loan rate R_t = r_t + s(g_t), plus
                median lines of R_t, r_t, and s(g_t).
    Right panel: histograms of R_t at selected times.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    N, nsim = sim["rates"].shape[0] - 1, sim["rates"].shape[1]
    time_grid = sim["time_grid"]
    dt = time_grid[1] - time_grid[0]

    # Pre‑allocate arrays for total rate and spread (annualised)
    R_new = np.full((N + 1, nsim), np.nan, dtype=np.float64)
    S     = np.full((N + 1, nsim), np.nan, dtype=np.float64)   # spread

    # Compute R_new and spread for each decision time
    for t in range(N):
        r_cur = sim["rates"][t]               # shape (nsim,)
        c_cur = sim["coupons"][t]
        k_cur = sim["capital_ratio"][t]

        # vectorised interpolation of optimal growth
        g = solver._interp3_uniform(
            solver.opt_g[t], r_cur, c_cur, k_cur,
            solver.r_grid, solver.c_grid, solver.k_grid
        )                                      # shape (nsim,)
        spread = solver.params.spread(g)       # annualised
        R_new[t + 1] = r_cur + spread
        S[t + 1]     = spread

    # ---- Left panel: quantile bands + component medians ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    t_plot = time_grid[1:]                     # skip t=0 because R_new is NaN there
    R_data = R_new[1:]
    r_data = sim["rates"][1:]                  # short rate at same times
    S_data = S[1:]

    # Quantiles of R_t
    q10 = np.quantile(R_data, 0.10, axis=1)
    q25 = np.quantile(R_data, 0.25, axis=1)
    q50 = np.quantile(R_data, 0.50, axis=1)
    q75 = np.quantile(R_data, 0.75, axis=1)
    q90 = np.quantile(R_data, 0.90, axis=1)

    axes[0].fill_between(t_plot, q10, q90, alpha=0.18, label="10–90% $R_t$")
    axes[0].fill_between(t_plot, q25, q75, alpha=0.35, label="25–75% $R_t$")
    axes[0].plot(t_plot, q50, linewidth=2, color="black", label="median $R_t$")

    # Median of the two components
    r_med = np.median(r_data, axis=1)
    s_med = np.median(S_data, axis=1)
    axes[0].plot(t_plot, r_med, linewidth=1.8, color="tab:blue", linestyle="--",
                 label="median $r_t$")
    axes[0].plot(t_plot, s_med, linewidth=1.8, color="tab:orange", linestyle="--",
                 label="median $s(g_t)$")

    axes[0].set_title("Loan rate $R_t = r_t + s(g_t)$ and its components")
    axes[0].set_xlabel("time (years)")
    axes[0].set_ylabel("rate (annualised)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # ---- Right panel: histograms of R_t ----
    time_steps = [int(round(ty / dt)) for ty in times_years if ty <= time_grid[-1]]
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for step, col, ty in zip(time_steps, colours, times_years):
        rates = R_new[step]                     # R_new at beginning of period 'step'
        axes[1].hist(rates, bins=30, alpha=0.5, label=f"t = {ty:.1f}y",
                     color=col, density=True)
    axes[1].set_title("Distribution of $R_t$ at selected times")
    axes[1].set_xlabel("rate")
    axes[1].set_ylabel("density")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.show()