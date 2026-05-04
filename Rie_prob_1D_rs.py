import weno_ext
import numpy as np
import matplotlib.pyplot as plt
import WENO_HLLC_2d
from plot_utils import interactive_plot_keyboard
grid_dict = {
    "nx":3,
    "ny":500,
    "dx":1/500,
}

control_dict = {
    "nstep": 100,
    "CFL": 0.3,
    "min_step_time": 1e-10,
    "max_time_step": 0.1,
    "file_storage": True
}

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

def init(grid_info,control_dict, gamma = 1.4):
    
    import datetime
    import os

    nx = grid_info["nx"]
    ny = grid_info["ny"]
    dx = grid_info["dx"]

    q = np.zeros((nx+6,ny+6,8))

    bound = int((nx+6)/2)

    q[:bound,:,4] = 1.0
    q[:bound, :, 5] = 0.0
    q[:bound, :, 6] = 0.0
    q[:bound, :, 7] = 1.0

    q[bound:, :, 4] = 0.125
    q[bound:, :, 5] = 0.0
    q[bound:, :, 6] = 0.0
    q[bound:, :, 7] = 0.1

    q[:, :, 0] = q[:, :, 4]
    q[:, :, 1] = q[:, :, 4]* q[:, :, 5]
    q[:, :, 2] = q[:, :, 4]* q[:, :, 6]
    q[:, :, 3] = q[:, :, 7]/(gamma-1) + 0.5* q[:,:,4]*(q[:,:,5]**2 + q[:,:,6]**2)

    q = apply_bc_corrected(q,grid_info)

    return q

def main(grid_info,control_dict, gamma = 1.4, file_storage = True):

    import datetime
    import os

    full_path = ""
    folder_name = ""

    nx = grid_info["nx"]
    ny = grid_info["ny"]
    dx = grid_info["dx"]

    q = np.zeros((nx+6,ny+6,8))

    bound = int((max(nx,ny)+6)/2)

    q[:,:bound, 4] = 1.0
    q[:,:bound, 5] = 0.0
    q[:,:bound, 6] = 0.0
    q[:,:bound, 7] = 1.0

    q[:,bound:, 4] = 0.125
    q[:,bound:, 5] = 0.0
    q[:,bound:, 6] = 0.0
    q[:,bound:, 7] = 0.1

    '''
    q[:bound,:,4] = 1.0
    q[:bound, :, 5] = 0.0
    q[:bound, :, 6] = 0.0
    q[:bound, :, 7] = 1.0

    q[bound:, :, 4] = 0.125
    q[bound:, :, 5] = 0.0
    q[bound:, :, 6] = 0.0
    q[bound:, :, 7] = 0.1
    '''

    q[:, :, 0] = q[:, :, 4]
    q[:, :, 1] = q[:, :, 4]* q[:, :, 5]
    q[:, :, 2] = q[:, :, 4]* q[:, :, 6]
    q[:, :, 3] = q[:, :, 7]/(gamma-1) + 0.5* q[:,:,4]*(q[:,:,5]**2 + q[:,:,6]**2)

    q = apply_bc_corrected(q,grid_info)

    if file_storage:
        current_time = datetime.datetime.now()
        folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        full_path = os.path.join("data", folder_name)
        os.makedirs(full_path,exist_ok=True)
        np.save(os.path.join(full_path,"0.npy"),q)

    t_list = np.array([0])

    T = 0

    for i in range(1,control_dict["nstep"]):
        print(i)
        speed = np.sqrt(q[3:nx+3,3:ny+3,5]**2+q[3:nx+3,3:ny+3,6]**2)
        c_max = np.max(np.sqrt(gamma*q[3:nx+3,3:ny+3,7]/np.maximum(q[3:nx+3,3:ny+3,0],1e-10))+speed)

        dt = control_dict["CFL"]*dx/c_max

        q = rk3(q,grid_dict,dt,gamma,force_hlle=True,jp_cri=(1,5))
        T += dt
        if file_storage:
            np.save(os.path.join(full_path,f"{i}.npy"), q)
        t_list = np.append(t_list, T)

    if file_storage:
        np.save(os.path.join(full_path,"time.npy"),t_list)
        interactive_plot_keyboard(full_path,initial_step=0,var='rho')



        

'''
field = init(grid_dict,control_dict)

q_l,f_l,q_r,f_r = weno_ext.weno_x_py(field,1.4)

flux = weno_ext.hllc_x_rs_py(q_l,q_r,f_l,f_r,1.4)
flux1 = WENO_HLLC_2d.HLLC_x(q_l,q_r,f_l,f_r,1.4)
res_next = weno_ext.l_local_py(field,grid_dict["dx"],1.4,False,(5,100))
res_next1 = WENO_HLLC_2d.L(field,grid_dict["dx"],1.4,False,(5.0,100))
print(res_next.shape)
for i in range(248,255):
    print("Rust Core:",res_next[i,2,0],"at",str(i))
    print("Python Core:",res_next1[i,2,0],"at",str(i))
print(field.shape)
plt.figure(figsize=(20,4))
plt.imshow(flux[:,:,0].T)
plt.show()


'''

main(grid_dict,control_dict)