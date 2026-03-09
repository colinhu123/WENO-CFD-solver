import weno_ext
import numpy as np
from plot_utils import interactive_plot_keyboard

grid_dict = {
    "nx": 200,
    "ny": 200,
    "dx": 1/200
}

control_dict = {
    "nstep": 100,
    "CFL": 0.1,
    "min_step_time": 1e-10,
    "max_time_step": 0.1,
    "file_storage":True,
    "force_hlle": False,
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

    for i in [0, 1, 2]:
        u[i, :, :] = u[3, :, :]

    for i in [nx+3, nx+4, nx+5]:
        u[i, :, :] = u[nx+2, :, :]
    for j in [0, 1, 2]:
        u[:, j, :] = u[:, 3, :]
    for j in [ny+3, ny+4, ny+5]:
        u[:, j, :] = u[:, ny+2, :]
    
    return u


def rk3(q,grid_dict,dt, gamma, force_hlle, jp_cri):
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
    q[quad_index:,:quad_index,5] = 0
    q[quad_index:,:quad_index,6] = 1
    q[quad_index:,:quad_index,7] = 0.25

    q[:quad_index,quad_index:,4] = 0.5
    q[:quad_index,quad_index:,5] = 1
    q[:quad_index,quad_index:,6] = 0
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

def main(grid_dict,control_dict,phys_dict):
    import datetime
    import os

    gamma = phys_dict["gamma"]
    nx = grid_dict["nx"]
    ny = grid_dict["ny"]
    q = init(grid_dict,control_dict,gamma)
    full_path = ""
    folder_name = ""

    gamma = 1.4
    dx = grid_dict["dx"]

    q = init(grid_dict, control_dict,gamma)

    file_storage = control_dict["file_storage"]

    nstep = control_dict["nstep"]
    
    if file_storage:
        current_time = datetime.datetime.now()
        folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        full_path = os.path.join("data", folder_name)
        os.makedirs(full_path,exist_ok=True)
        np.save(os.path.join(full_path,"0.npy"),q)
    t_list = np.array([0])
    T = 0
    for i in range(1,nstep):
        print(i)
        speed = np.sqrt(q[3:nx+3,3:ny+3,5]**2+q[3:nx+3,3:ny+3,6]**2)
        c_max = np.max(np.sqrt(gamma*q[3:nx+3,3:ny+3,7]/np.maximum(q[3:nx+3,3:ny+3,0],1e-10))+speed)

        dt = control_dict["CFL"]*dx/c_max

        q = rk3(q,grid_dict,dt,gamma,force_hlle=False,jp_cri=(1,5))
        T += dt
        if file_storage:
            np.save(os.path.join(full_path,f"{i}.npy"), q)
        t_list = np.append(t_list, T)
        
        t_list = np.append(t_list,T)
    if file_storage:
        np.save(os.path.join(full_path,"time.npy"),t_list)
        interactive_plot_keyboard(full_path,initial_step=0,var='rho')



main(grid_dict,control_dict,phys_dict)
