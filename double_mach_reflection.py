import weno_ext
import numpy as np
from plot_utils import interactive_plot_keyboard
import unitest

import matplotlib.pyplot as plt
grid_dict = {
    "nx":400,
    "ny":100,
    "dx":1/100,
    "x":4,
    "y":1
}

control_dict = {
    "nstep": 800,
    "CFL": 0.1,
    "max_time_step": 0.1,
    "file_storage":True,
    "force_hlle": False,
    "visualize": True,
    "mode":"debug",
    "t_final":0.2,
    "jp_cri":(10000,11000)
}

phys_dict = {
    "gamma":1.4
}

def apply_bc_DMR(u, grid_info):
    nx = grid_info.get("nx")
    ny = grid_info.get("ny")
    bound = int(np.floor(1/6*nx/grid_info["x"]+2))
    post_shock = [8,4.125,-7.144,116.5]
    for i in [0, 1, 2]:
        u[i, :, 4:] = post_shock

    for i in [nx+3, nx+4, nx+5]:
        u[i, :, :] = u[nx+2, :, :]
    for j in [0, 1, 2]:
        u[:bound, j, 4] = post_shock[0]
        u[:bound, j, 5] = post_shock[1]
        u[:bound, j, 6] = post_shock[2]
        u[:bound, j, 7] = post_shock[3]

        u[bound:, j, :] = u[bound:, 5-j, :]
        u[bound:, j, 2] = -u[bound:, 5-j, 2]
        u[bound:, j, 6] = u[bound:, 5-j, 6]
    for j in [ny+3, ny+4, ny+5]:
        u[:, j, :] = u[:, ny+2, :]
    
    return u



def rk3(q,grid_dict,dt, gamma, force_hlle,apply_bc, jp_cri=(0.2,1.0)):
    nx = grid_dict.get("nx")
    ny = grid_dict.get("ny")
    dx = grid_dict.get("dx")

    l0 = weno_ext.l_local_py(q,dx,gamma,force_hlle, jp_cri)

    q1 = q.copy()
    q1[3: nx+3, 3:ny+3,:4] = q[3: nx+3, 3:ny+3,:4] + dt*l0

    q1 = apply_bc(q1,grid_dict)

    l1 = weno_ext.l_local_py(q1,dx,gamma,force_hlle,jp_cri)
    q2 = q.copy()
    q2[3: nx+3, 3:ny+3,:4] = (
        0.75*q[3: nx+3, 3:ny+3,:4] + 
        0.25*q1[3: nx+3, 3:ny+3,:4] + 
        0.25*dt*l1
    )
    q2 = apply_bc(q2,grid_dict)

    l2 = weno_ext.l_local_py(q2,dx,gamma,force_hlle,jp_cri)
    q_next = q.copy()
    q_next[3: nx+3, 3:ny+3,:4] = (
        1/3 * q[3: nx+3, 3:ny+3,:4] + 
        2/3 * q2[3: nx+3, 3:ny+3,:4] +
        2/3 * dt*l2
    )


    q_next = apply_bc(q_next,grid_dict)
    (p, a, rho_safe, u, v, h) = weno_ext.con2primi_py(q_next,gamma)
    q_next[:,:,4] = rho_safe
    q_next[:,:,5] = u
    q_next[:,:,6] = v
    q_next[:,:,7] = p

    return apply_bc(q_next,grid_dict)


def rk3_refac(q,grid_dict,dt, gamma, force_hlle,apply_bc,bound, jp_cri=(0.2,1.0)):
    nx = grid_dict.get("nx")
    ny = grid_dict.get("ny")
    dx = grid_dict.get("dx")

    l0 = weno_ext.l_local_py(q,dx,gamma,force_hlle, jp_cri)

    q1 = q.copy()
    q1[3: nx+3, 3:ny+3,:4] = q[3: nx+3, 3:ny+3,:4] + dt*l0

    q1 = apply_bc(q1,bound,grid_dict)

    l1 = weno_ext.l_local_py(q1,dx,gamma,force_hlle,jp_cri)
    q2 = q.copy()
    q2[3: nx+3, 3:ny+3,:4] = (
        0.75*q[3: nx+3, 3:ny+3,:4] + 
        0.25*q1[3: nx+3, 3:ny+3,:4] + 
        0.25*dt*l1
    )
    q2 = apply_bc(q2,bound,grid_dict)

    l2 = weno_ext.l_local_py(q2,dx,gamma,force_hlle,jp_cri)
    q_next = q.copy()
    q_next[3: nx+3, 3:ny+3,:4] = (
        1/3 * q[3: nx+3, 3:ny+3,:4] + 
        2/3 * q2[3: nx+3, 3:ny+3,:4] +
        2/3 * dt*l2
    )


    q_next = apply_bc(q_next,bound,grid_dict)
    (p, a, rho_safe, u, v, h) = weno_ext.con2primi_py(q_next,gamma)
    q_next[:,:,4] = rho_safe
    q_next[:,:,5] = u
    q_next[:,:,6] = v
    q_next[:,:,7] = p

    return apply_bc(q_next,bound,grid_dict)

def ind2phy(i,j,grid_dict):
    x_pos = (i-3+0.5)/grid_dict["nx"]*grid_dict["x"]
    y_pos = (j-3+0.5)/grid_dict["ny"]*grid_dict["y"]

    return (x_pos,y_pos)

def init(grid_dict,control_dict, gamma = 1.4):
    nx = grid_dict["nx"]
    ny = grid_dict["ny"]
    dx = grid_dict["dx"]

    q = np.zeros((nx+6,ny+6,8))

    q[:,:,4] = 1.4
    q[:,:,5] = 0
    q[:,:,6] = 0
    q[:,:,7] = 1

    def shock(x):
        return np.sqrt(3)*x - np.sqrt(3)/6
    
    for i in range(nx+6):
        for j in range(ny+6):
            (x,y) = ind2phy(i,j,grid_dict)
            if y-shock(x) >= 0:
                q[i,j,4] = 8
                q[i,j,5] = 4.125
                q[i,j,6] = -7.1447
                q[i,j,7] = 116.5


    q[:, :, 0] = q[:, :, 4]
    q[:, :, 1] = q[:, :, 4]* q[:, :, 5]
    q[:, :, 2] = q[:, :, 4]* q[:, :, 6]
    q[:, :, 3] = q[:, :, 7]/(gamma-1) + 0.5* q[:,:,4]*(q[:,:,5]**2 + q[:,:,6]**2)

    q = apply_bc_DMR(q,grid_dict)

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
    q,bound = unitest.init_refac(grid_dict, control_dict, gamma)

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
                q = rk3_refac(q, grid_dict, dt, gamma, 
                        force_hlle=control_dict.get("force_hlle", False),
                        apply_bc=unitest.apply_bc_refac, 
                        bound=bound,
                        jp_cri=jp_cri)
            else:
                q = rk3_refac(q, grid_dict, dt, gamma,
                        force_hlle=control_dict.get("force_hlle", False),
                        bound = bound,
                        apply_bc=unitest.apply_bc_refac)

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
                q = rk3_refac(q, grid_dict, dt, gamma, 
                        force_hlle=control_dict.get("force_hlle", False), 
                        apply_bc=unitest.apply_bc_refac,
                        bound = bound,
                        jp_cri=jp_cri)
            else:
                q = rk3_refac(q, grid_dict, dt, gamma, 
                        force_hlle=control_dict.get("force_hlle", False),
                        bound = bound,
                        apply_bc=unitest.apply_bc)

            T += dt
            step += 1

            if file_storage:
                np.save(os.path.join(full_path, f"{step}.npy"), q)
            t_list.append(T)

    # finished time loop
    if file_storage:
        np.save(os.path.join(full_path, "time.npy"), np.array(t_list))
        if control_dict.get("visualize", False):
            interactive_plot_keyboard(full_path, initial_step=0, var='rho',ghost=3)

    if control_dict.get("mode", "") == "opt":
        return q

    # otherwise return nothing (or you can return diagnostics if desired)
    return None


if __name__ == "__main__":
    main_rs(grid_dict,control_dict,phys_dict)

