import weno_ext
import numpy as np
from plot_utils import interactive_plot_keyboard

grid_dict = {
    "nx": 200,
    "ny": 200,
    "dx": 1/200
}

control_dict = {
    "nstep": 150,
    "CFL": 0.1,
    "min_step_time": 1e-10,
    "max_time_step": 0.1,
    "file_storage": True,
    "force_hlle": False,
    "visualize": True,
    "mode": "opt",
    #"t_final": 0.05,
    "jp_cri": (0.1, 12)
}

phys_dict = {
    "gamma": 1.4
}

# ---------------------------------------------------------------------------
# Sign masks for reflecting BC — shape (8,)
# q layout: [rho, rho*u, rho*v, rho*e, rho, u, v, p]
#   idx:       0     1      2      3     4   5  6  7
#
# y-wall (bottom / top):   normal = y  →  flip rho*v (2) and v (6)
# x-wall (left  / right):  normal = x  →  flip rho*u (1) and u (5)
# corner:                  both walls  →  flip both pairs
# ---------------------------------------------------------------------------
_S_Y = np.array([ 1,  1, -1,  1,  1,  1, -1,  1], dtype=np.float64)  # y-wall
_S_X = np.array([ 1, -1,  1,  1,  1, -1,  1,  1], dtype=np.float64)  # x-wall
_S_C = np.array([ 1, -1, -1,  1,  1, -1, -1,  1], dtype=np.float64)  # corner


def apply_bc_reflecting(u: np.ndarray, grid_info: dict) -> np.ndarray:
    """
    Reflecting (mirror) boundary conditions with 3 ghost cells on each side.

    u : shape (nx+6, ny+6, 8)
        [rho, rho*u, rho*v, rho*e, rho, u, v, p]

    Strategy
    --------
    * Fill the four edges EXCLUDING their corner columns/rows so that no
      edge pass overwrites another edge's ghost cells.
    * Fill the four 3×3 corner blocks explicitly afterwards, each with the
      combined flip mask _S_C (both normal components negated).

    Returns u (same array, modified in-place and returned for convenience).
    """
    ng = 3
    m, n = u.shape[0], u.shape[1]

    # ------------------------------------------------------------------ #
    #  BOTTOM  (ghost rows 0,1,2  ←  interior rows 5,4,3)                #
    #  normal = y  →  flip rho*v (idx 2) and v (idx 6)                   #
    #  operate only on interior columns  [ng : n-ng]                     #
    # ------------------------------------------------------------------ #
    for g in range(ng):
        mirror = 2 * ng - 1 - g          # 0→5, 1→4, 2→3
        u[g, ng:n-ng, :] = u[mirror, ng:n-ng, :] * _S_Y

    # ------------------------------------------------------------------ #
    #  TOP  (ghost rows m-3,m-2,m-1  ←  interior rows m-6,m-5,m-4)      #
    # ------------------------------------------------------------------ #
    for g in range(ng):
        interior = m - 2 * ng + g        # m-6, m-5, m-4
        ghost    = m - ng + g            # m-3, m-2, m-1
        u[ghost, ng:n-ng, :] = u[interior, ng:n-ng, :] * _S_Y

    # ------------------------------------------------------------------ #
    #  LEFT  (ghost cols 0,1,2  ←  interior cols 5,4,3)                  #
    #  normal = x  →  flip rho*u (idx 1) and u (idx 5)                   #
    #  operate only on interior rows  [ng : m-ng]                        #
    # ------------------------------------------------------------------ #
    for g in range(ng):
        mirror = 2 * ng - 1 - g
        u[ng:m-ng, g, :] = u[ng:m-ng, mirror, :] * _S_X

    # ------------------------------------------------------------------ #
    #  RIGHT  (ghost cols n-3,n-2,n-1  ←  interior cols n-6,n-5,n-4)    #
    # ------------------------------------------------------------------ #
    for g in range(ng):
        interior = n - 2 * ng + g
        ghost    = n - ng + g
        u[ng:m-ng, ghost, :] = u[ng:m-ng, interior, :] * _S_X

    # ------------------------------------------------------------------ #
    #  CORNERS  — filled once with combined flip mask _S_C               #
    #  Each 3×3 corner block mirrors from the matching interior block.   #
    # ------------------------------------------------------------------ #
    for gi in range(ng):
        for gj in range(ng):
            mi = 2 * ng - 1 - gi          # mirror row index
            mj = 2 * ng - 1 - gj          # mirror col index

            # bottom-left
            u[gi, gj, :] = u[mi, mj, :] * _S_C

            # bottom-right
            int_j = n - 2 * ng + gj
            gh_j  = n - ng + gj
            u[gi, gh_j, :] = u[mi, int_j, :] * _S_C

            # top-left
            int_i = m - 2 * ng + gi
            gh_i  = m - ng + gi
            u[gh_i, gj, :] = u[int_i, mj, :] * _S_C

            # top-right
            u[gh_i, gh_j, :] = u[int_i, int_j, :] * _S_C

    return u


# ---------------------------------------------------------------------------
# Helper: sync primitive variables from conserved (indices 4-7)
# ---------------------------------------------------------------------------
def _sync_primitives(q: np.ndarray, gamma: float) -> np.ndarray:
    p, a, rho, u, v, h = weno_ext.con2primi_py(q, gamma)
    q[:, :, 4] = rho
    q[:, :, 5] = u
    q[:, :, 6] = v
    q[:, :, 7] = p
    return q


# ---------------------------------------------------------------------------
# RK3 (SSP) time integrator
# ---------------------------------------------------------------------------
def rk3_cc(q: np.ndarray, grid_dict: dict, dt: float,
           gamma: float, force_hlle: bool,
           jp_cri: tuple = (0.2, 1.0)) -> np.ndarray:

    nx, ny, dx = grid_dict["nx"], grid_dict["ny"], grid_dict["dx"]

    sl = np.s_[3:nx+3, 3:ny+3, :4]   # interior slice, conserved only

    # ---- Stage 1 -------------------------------------------------------
    apply_bc_reflecting(q, grid_dict)
    l0 = weno_ext.l_local_py(q, dx, gamma, force_hlle, jp_cri)

    q1 = q.copy()
    q1[sl] = q[sl] + dt * l0
    apply_bc_reflecting(q1, grid_dict)
    _sync_primitives(q1, gamma)
    apply_bc_reflecting(q1, grid_dict)   # re-fill after primitive sync

    # ---- Stage 2 -------------------------------------------------------
    l1 = weno_ext.l_local_py(q1, dx, gamma, force_hlle, jp_cri)

    q2 = q.copy()
    q2[sl] = 0.75 * q[sl] + 0.25 * q1[sl] + 0.25 * dt * l1
    apply_bc_reflecting(q2, grid_dict)
    _sync_primitives(q2, gamma)
    apply_bc_reflecting(q2, grid_dict)

    # ---- Stage 3 -------------------------------------------------------
    l2 = weno_ext.l_local_py(q2, dx, gamma, force_hlle, jp_cri)

    q_next = q.copy()
    q_next[sl] = (1/3) * q[sl] + (2/3) * q2[sl] + (2/3) * dt * l2
    apply_bc_reflecting(q_next, grid_dict)
    _sync_primitives(q_next, gamma)
    apply_bc_reflecting(q_next, grid_dict)

    return q_next


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init(grid_dict: dict, control_dict: dict, gamma: float = 1.4) -> np.ndarray:

    nx = grid_dict["nx"]
    ny = grid_dict["ny"]

    q = np.zeros((nx + 6, ny + 6, 8))

    # quad_index is the cell index of the quadrant boundary in the full
    # (ghost-padded) array.  For a symmetric nx==ny grid this sits at
    # the physical midpoint:  3 (ghost) + nx//2
    quad_index = 3 + nx // 2

    # Primitive variables per quadrant  [rho, u, v, p]  → indices 4,5,6,7
    # bottom-left
    q[:quad_index, :quad_index, 4] = 0.125
    q[:quad_index, :quad_index, 5] = 1.0
    q[:quad_index, :quad_index, 6] = 1.0
    q[:quad_index, :quad_index, 7] = 0.125

    # top-left  (i >= quad_index, j < quad_index)
    q[quad_index:, :quad_index, 4] = 0.5
    q[quad_index:, :quad_index, 5] = 1.0
    q[quad_index:, :quad_index, 6] = 0.0
    q[quad_index:, :quad_index, 7] = 0.25

    # bottom-right  (i < quad_index, j >= quad_index)
    q[:quad_index, quad_index:, 4] = 0.5
    q[:quad_index, quad_index:, 5] = 0.0
    q[:quad_index, quad_index:, 6] = 1.0
    q[:quad_index, quad_index:, 7] = 0.25

    # top-right
    q[quad_index:, quad_index:, 4] = 1.0
    q[quad_index:, quad_index:, 5] = 0.0
    q[quad_index:, quad_index:, 6] = 0.0
    q[quad_index:, quad_index:, 7] = 1.0

    # Conserved variables from primitives
    rho = q[:, :, 4]
    u   = q[:, :, 5]
    v   = q[:, :, 6]
    p   = q[:, :, 7]

    q[:, :, 0] = rho
    q[:, :, 1] = rho * u
    q[:, :, 2] = rho * v
    q[:, :, 3] = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    apply_bc_reflecting(q, grid_dict)

    return q


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main_rs(grid_dict: dict, control_dict: dict, phys_dict: dict):
    import datetime
    import os
    from tqdm import tqdm

    gamma      = phys_dict.get("gamma", 1.4)
    nx         = int(grid_dict["nx"])
    ny         = int(grid_dict["ny"])
    dx         = float(grid_dict.get("dx", 1.0))
    file_storage = bool(control_dict.get("file_storage", False))
    force_hlle   = bool(control_dict.get("force_hlle", False))
    jp_cri       = control_dict.get("jp_cri", (0.2, 1.0))

    t_final    = control_dict.get("t_final", None)
    nstep_user = control_dict.get("nstep", None)

    if t_final is not None and nstep_user is not None:
        print("Info: both t_final and nstep provided — t_final is primary, nstep is safety cap.")
    elif t_final is not None:
        print("Info: using t_final.")
    elif nstep_user is not None:
        print("Info: using nstep.")
    else:
        raise ValueError("Either 't_final' or 'nstep' must be provided.")

    max_steps = int(nstep_user) if nstep_user is not None else int(1e8)

    q = init(grid_dict, control_dict, gamma)

    full_path = ""
    if file_storage:
        folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        full_path   = os.path.join("data", folder_name)
        os.makedirs(full_path, exist_ok=True)
        np.save(os.path.join(full_path, "0.npy"), q)

    t_list   = [0.0]
    T        = 0.0
    step     = 0
    eps_time = 1e-12

    def _cfl_dt(q, remaining=None):
        speed = np.sqrt(q[3:3+nx, 3:3+ny, 5]**2 + q[3:3+nx, 3:3+ny, 6]**2)
        c     = np.sqrt(gamma * q[3:3+nx, 3:3+ny, 7]
                        / np.maximum(q[3:3+nx, 3:3+ny, 0], 1e-14))
        c_max = float(np.max(c + speed)) + 1e-16
        dt    = float(control_dict.get("CFL", 0.5)) * dx / c_max
        if remaining is not None:
            dt = min(dt, remaining)
        return dt

    def _step(q, dt):
        return rk3_cc(q, grid_dict, dt, gamma,
                      force_hlle=force_hlle, jp_cri=jp_cri)

    # ------------------------------------------------------------------ #
    #  TIME-CONTROLLED RUN                                                #
    # ------------------------------------------------------------------ #
    if t_final is not None:
        pbar = tqdm(
            total=t_final,
            desc="simulating",
            unit="t",
        )
        pbar.set_postfix(dt=0.0, step=0)
        while (T < t_final - eps_time) and (step < max_steps):
            dt = _cfl_dt(q, remaining=t_final - T)
            q  = _step(q, dt)
            T += dt; step += 1
            pbar.n = min(T, t_final)
            pbar.set_postfix(dt=dt, step=step, refresh=False)
            pbar.refresh()
            if file_storage:
                np.save(os.path.join(full_path, f"{step}.npy"), q)
            t_list.append(T)
        pbar.close()

    # ------------------------------------------------------------------ #
    #  STEP-CONTROLLED RUN                                                #
    # ------------------------------------------------------------------ #
    else:
        nstep = int(nstep_user)
        pbar  = tqdm(
            range(1, nstep + 1),
            desc="simulating",
            unit="step",
        )
        pbar.set_postfix(dt=0.0, t=0.0)
        for _ in pbar:
            dt = _cfl_dt(q)
            q  = _step(q, dt)
            T += dt; step += 1
            pbar.set_postfix(dt=dt, t=T, refresh=False)
            if file_storage:
                np.save(os.path.join(full_path, f"{step}.npy"), q)
            t_list.append(T)

    # ------------------------------------------------------------------ #
    if file_storage:
        np.save(os.path.join(full_path, "time.npy"), np.array(t_list))
        if control_dict.get("visualize", False):
            interactive_plot_keyboard(full_path, initial_step=0, var='rho')

    if control_dict.get("mode", "") == "opt":
        return q

    return None


if __name__ == "__main__":
    main_rs(grid_dict, control_dict, phys_dict)