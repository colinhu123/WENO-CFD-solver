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
    "file_storage":True,
    "force_hlle": False,
    "visualize": True,
    "mode":"opt",
    #"t_final":0.05,
    "jp_cri":(1,5)
}

phys_dict = {
    "gamma":1.4
}

def apply_bc_corrected(u, grid_info):
    nx = grid_info.get("nx")
    ny = grid_info.get("ny")

    i_start, i_end = 3, 3 + nx
    j_start, j_end = 3, 3 + ny

    for g in range(3):
        mirror = 5 - g          # g=0->5, g=1->4, g=2->3
        u[g, :, 0] =  u[mirror, :, 0]   # rho
        u[g, :, 1] =  -u[mirror, :, 1]   # rho*u — flip (normal to bottom wall is y... 
        u[g, :, 2] =  u[mirror, :, 2]   # rho*v — tangential
        u[g, :, 3] =  u[mirror, :, 3]   # E

    # --- TOP boundary (ghost rows m-3,m-2,m-1 — mirror from m-6,m-5,m-4) ---
    m = u.shape[0]
    for g in range(3):
        interior = m - 6 + g    # m-6, m-5, m-4
        ghost    = m - 3 + g    # m-3, m-2, m-1
        u[ghost, :, 0] =  u[interior, :, 0]
        u[ghost, :, 1] =  -u[interior, :, 1]   # rho*u — flip
        u[ghost, :, 2] =  u[interior, :, 2]   # rho*v — tangential
        u[ghost, :, 3] =  u[interior, :, 3]

    # --- LEFT boundary (ghost cols 0,1,2 — mirror from cols 5,4,3) ---
    for g in range(3):
        mirror = 5 - g
        u[:, g, 0] =  u[:, mirror, 0]
        u[:, g, 1] =  u[:, mirror, 1]   # rho*u — tangential
        u[:, g, 2] = -u[:, mirror, 2]   # rho*v — flip (normal to left wall is x)
        u[:, g, 3] =  u[:, mirror, 3]

    # --- RIGHT boundary (ghost cols n-3,n-2,n-1 — mirror from n-6,n-5,n-4) ---
    n = u.shape[1]
    for g in range(3):
        interior = n - 6 + g
        ghost    = n - 3 + g
        u[:, ghost, 0] =  u[:, interior, 0]
        u[:, ghost, 1] =  u[:, interior, 1]   # rho*u — tangential
        u[:, ghost, 2] = -u[:, interior, 2]   # rho*v — flip
        u[:, ghost, 3] =  u[:, interior, 3]
    
    return u


def rk3(q,grid_dict,dt, gamma, force_hlle, jp_cri=(0.2,1.0)):
    nx = grid_dict.get("nx")
    ny = grid_dict.get("ny")
    dx = grid_dict.get("dx")

    l0 = weno_ext.l_local_py(q,dx,gamma,force_hlle, jp_cri)

    q1 = q.copy()
    q1[3: nx+3, 3:ny+3,:4] = q[3: nx+3, 3:ny+3,:4] + dt*l0

    q1 = apply_bc_corrected(q1,grid_dict)

    l1 = weno_ext.l_local_py(q1,dx,gamma,force_hlle,jp_cri)
    q2 = q.copy()
    q2[3: nx+3, 3:ny+3,:4] = (
        0.75*q[3: nx+3, 3:ny+3,:4] + 
        0.25*q1[3: nx+3, 3:ny+3,:4] + 
        0.25*dt*l1
    )
    q2 = apply_bc_corrected(q2,grid_dict)

    l2 = weno_ext.l_local_py(q2,dx,gamma,force_hlle,jp_cri)
    q_next = q.copy()
    q_next[3: nx+3, 3:ny+3,:4] = (
        1/3 * q[3: nx+3, 3:ny+3,:4] + 
        2/3 * q2[3: nx+3, 3:ny+3,:4] +
        2/3 * dt*l2
    )


    q_next = apply_bc_corrected(q_next,grid_dict)
    (p, a, rho_safe, u, v, h) = weno_ext.con2primi_py(q_next,gamma)
    q_next[:,:,4] = rho_safe
    q_next[:,:,5] = u
    q_next[:,:,6] = v
    q_next[:,:,7] = p

    return apply_bc_corrected(q_next,grid_dict)

def rk3_cc(q, grid_dict, dt, gamma, force_hlle, jp_cri=(0.2, 1.0)):
    nx, ny, dx = grid_dict["nx"], grid_dict["ny"], grid_dict["dx"]

    # Stage 1
    apply_bc_corrected(q,grid_dict)
    l0 = weno_ext.l_local_py(q, dx, gamma, force_hlle, jp_cri)
    q1 = q.copy()
    q1[3:nx+3, 3:ny+3, :4] = q[3:nx+3, 3:ny+3, :4] + dt * l0
    q1 = apply_bc_corrected(q1, grid_dict)
    # ← sync primitives on q1 before next L() call
    p, a, rho, u, v, h = weno_ext.con2primi_py(q1, gamma)
    q1[:,:,4]=rho; q1[:,:,5]=u; q1[:,:,6]=v; q1[:,:,7]=p

    # Stage 2
    q1 = apply_bc_corrected(q1,grid_dict)
    l1 = weno_ext.l_local_py(q1, dx, gamma, force_hlle, jp_cri)
    q2 = q.copy()
    q2[3:nx+3, 3:ny+3, :4] = (0.75*q[3:nx+3, 3:ny+3, :4]
                             + 0.25*q1[3:nx+3, 3:ny+3, :4]
                             + 0.25*dt*l1)
    q2 = apply_bc_corrected(q2, grid_dict)
    # ← sync primitives on q2 before next L() call
    p, a, rho, u, v, h = weno_ext.con2primi_py(q2, gamma)
    q2[:,:,4]=rho; q2[:,:,5]=u; q2[:,:,6]=v; q2[:,:,7]=p

    # Stage 3
    q2 = apply_bc_corrected(q2,grid_dict)
    l2 = weno_ext.l_local_py(q2, dx, gamma, force_hlle, jp_cri)
    q_next = q.copy()
    q_next[3:nx+3, 3:ny+3, :4] = (1/3*q[3:nx+3, 3:ny+3, :4]
                                 + 2/3*q2[3:nx+3, 3:ny+3, :4]
                                 + 2/3*dt*l2)
    q_next = apply_bc_corrected(q_next, grid_dict)
    # final sync (already in your code)
    p, a, rho, u, v, h = weno_ext.con2primi_py(q_next, gamma)
    q_next[:,:,4]=rho; q_next[:,:,5]=u; q_next[:,:,6]=v; q_next[:,:,7]=p

    return apply_bc_corrected(q_next, grid_dict)

def init(grid_dict,control_dict, gamma = 1.4):

    nx = grid_dict["nx"]
    ny = grid_dict["ny"]
    dx = grid_dict["dx"]

    q = np.zeros((nx+6,ny+6,8))

    quad_index = int(((nx+ny)/2+6)/2)

    q[:quad_index,:quad_index,4] = 0.125
    q[:quad_index,:quad_index,5] = 1
    q[:quad_index,:quad_index,6] = 1
    q[:quad_index,:quad_index,7] = 0.125

    q[quad_index:,:quad_index,4] = 0.5
    q[quad_index:,:quad_index,5] = 1
    q[quad_index:,:quad_index,6] = 0
    q[quad_index:,:quad_index,7] = 0.25

    q[:quad_index,quad_index:,4] = 0.5
    q[:quad_index,quad_index:,5] = 0
    q[:quad_index,quad_index:,6] = 1
    q[:quad_index,quad_index:,7] = 0.25

    q[quad_index:,quad_index:,4] = 1
    q[quad_index:,quad_index:,5] = 0
    q[quad_index:,quad_index:,6] = 0
    q[quad_index:,quad_index:,7] = 1


    q[:, :, 0] = q[:, :, 4]
    q[:, :, 1] = q[:, :, 4]* q[:, :, 5]
    q[:, :, 2] = q[:, :, 4]* q[:, :, 6]
    q[:, :, 3] = q[:, :, 7]/(gamma-1) + 0.5* q[:,:,4]*(q[:,:,5]**2 + q[:,:,6]**2)

    q = apply_bc_corrected(q,grid_dict)

    return q

def main_rs(grid_dict, control_dict, phys_dict):
    import datetime
    import os
    import numpy as np

    gamma = phys_dict.get("gamma", 1.4)
    nx = int(grid_dict["nx"])
    ny = int(grid_dict["ny"])
    dx = float(grid_dict.get("dx", 1.0))

    # initialize solution
    q = init(grid_dict, control_dict, gamma)

    # file storage settings
    file_storage = bool(control_dict.get("file_storage", False))

    # time/step control: t_final takes priority if provided
    t_final = control_dict.get("t_final", None)   # physical end time (preferred)
    nstep_user = control_dict.get("nstep", None)  # max steps or fixed-step fallback

    if t_final is not None:
        print("Info: using t_final, ignoring nstep if provided.")
    elif nstep_user is not None:
        print("Info: using nstep (no t_final specified).")
    else:
        raise ValueError("Either 't_final' or 'nstep' must be provided.")

    if (t_final is not None) and (nstep_user is not None):
        print("Info: both t_final and nstep provided — using t_final as primary; nstep will be used as a safety cap.")

    # safety cap on steps: if nstep_user provided use it, else use a large cap
    max_steps = int(nstep_user) if nstep_user is not None else int(1e8)

    # output folder
    full_path = ""
    if file_storage:
        current_time = datetime.datetime.now()
        folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        full_path = os.path.join("data", folder_name)
        os.makedirs(full_path, exist_ok=True)
        np.save(os.path.join(full_path, "0.npy"), q)

    # time-stepping loop
    t_list = [0.0]
    T = 0.0
    step = 0
    eps_time = 1e-12

    # get jp_cri if user specified (fallback to default inside rk3 if not)
    jp_cri = control_dict.get("jp_cri", None)

    # main loop: if t_final provided run by time, else run by step count
    if t_final is not None:
        # time-controlled run (adaptive dt)
        while (T < t_final - eps_time) and (step < max_steps):
            # compute max wave speed on interior (indices 3:3+nx, 3:3+ny)
            speed = np.sqrt(q[3:3+nx, 3:3+ny, 5]**2 + q[3:3+nx, 3:3+ny, 6]**2)
            c = np.sqrt(gamma * q[3:3+nx, 3:3+ny, 7] / np.maximum(q[3:3+nx, 3:3+ny, 0], 1e-14))
            c_max = float(np.max(c + speed))

            dt_cfl = float(control_dict.get("CFL", 0.5)) * dx / (c_max + 1e-16)
            remaining = t_final - T
            # ensure dt doesn't overshoot final time
            dt = min(dt_cfl, remaining)

            # call rk3 with user's jp_cri if provided, else default inside rk3
            if jp_cri is not None:
                q = rk3_cc(q, grid_dict, dt, gamma, force_hlle=control_dict.get("force_hlle", False), jp_cri=jp_cri)
            else:
                q = rk3_cc(q, grid_dict, dt, gamma, force_hlle=control_dict.get("force_hlle", False))

            T += dt
            step += 1

            if file_storage:
                np.save(os.path.join(full_path, f"{step}.npy"), q)
            t_list.append(T)
    else:
        # step-controlled run (fixed number of steps but with adaptive dt)
        if nstep_user is None:
            raise ValueError("Either 't_final' or 'nstep' must be provided in control_dict.")
        # run for exactly nstep_user steps (adaptive dt each step)
        nstep = int(nstep_user)
        for i in range(1, nstep + 1):
            speed = np.sqrt(q[3:3+nx, 3:3+ny, 5]**2 + q[3:3+nx, 3:3+ny, 6]**2)
            c = np.sqrt(gamma * q[3:3+nx, 3:3+ny, 7] / np.maximum(q[3:3+nx, 3:3+ny, 0], 1e-14))
            c_max = float(np.max(c + speed))

            dt = float(control_dict.get("CFL", 0.5)) * dx / (c_max + 1e-16)

            if jp_cri is not None:
                q = rk3_cc(q, grid_dict, dt, gamma, force_hlle=control_dict.get("force_hlle", False), jp_cri=jp_cri)
            else:
                q = rk3_cc(q, grid_dict, dt, gamma, force_hlle=control_dict.get("force_hlle", False))

            T += dt
            step += 1

            if file_storage:
                np.save(os.path.join(full_path, f"{step}.npy"), q)
            t_list.append(T)

    # finished time loop
    if file_storage:
        np.save(os.path.join(full_path, "time.npy"), np.array(t_list))
        if control_dict.get("visualize", False):
            interactive_plot_keyboard(full_path, initial_step=0, var='rho')

    if control_dict.get("mode", "") == "opt":
        return q

    # otherwise return nothing (or you can return diagnostics if desired)
    return None


if __name__ == "__main__":
    main_rs(grid_dict,control_dict,phys_dict)
