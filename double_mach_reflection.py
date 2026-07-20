import weno_ext
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import interactive_plot_keyboard

grid_dict = {
    "nx": 400,
    "ny": 100,
    "dx": 1/100      # ← 确认已修复
}

control_dict = {
    "nstep": 100,
    "CFL": 0.3,
    "min_step_time": 1e-10,
    "max_time_step": 0.1,
    "file_storage": True,
    "force_hlle": False,
    "visualize": True,
    "mode": "opt",
    "jp_cri": (0.5, 2),   # ← 从这里开始调
    "weno type": True
}

phys_dict = {
    "gamma": 1.4
}

def apply_bc(u):
    ng = 3
    nx = u.shape[0]
    ny = u.shape[1]
    g = 0
    for g in range(ng):
        interior = ny - 2*ng +g
        ghost = ny - 1 - g
        u[:,ghost,:] = u[:,interior, :]
    g = 0
    for  g in range(ng):
        interior = nx -2*ng + g
        ghost = nx - 1 - g
        u[ghost,:,:] = u[interior, :,:]
    #right and top boundary condition

    pass

def _sync_primitives(q: np.ndarray, gamma: float) -> np.ndarray:
    p, a, rho, u, v, h = weno_ext.con2primi_py(q, gamma)
    q[:, :, 4] = rho
    q[:, :, 5] = u
    q[:, :, 6] = v
    q[:, :, 7] = p
    return q

def ind2phy(i,j,grid_dict):
    dx = grid_dict.get("dx")
    return ((i+0.5)*dx,(j+0.5)*dx)

def shock(x,y):
    if y - np.sqrt(3)*x + np.sqrt(3)/6 > 0:
        return True
    else:
        return False
    

def init(grid_fic, control_dict, gamma):

    nx = grid_dict["nx"]
    ny = grid_dict["ny"]

    q = np.zeros((nx + 6,ny + 6, 8))
    q[:,:,4] = 1
    q[:, :, 5] = 0
    q[:, :, 6] = 0
    q[:, :, 7] = 1.4

    for i in range(nx):
        for j in range(ny):
            (x,y) = ind2phy(i,j,grid_dict)
            
            if shock(x,y):
                q[i+3,j+3,4] = 8
                q[i+3,j+3,5] = 4.125
                q[i+3,j+3,6] = -7.1447
                q[i+3,j+3, 7] = 116.5
            else:
                continue

    q = _sync_primitives(q,1.4)
    return q


plt.imshow(init(grid_dict,control_dict,gamma = 1.4)[:,:,4])
plt.show()


'''
This case is unsolved since the boundary conditions seem like full of problems 
and require further inspection
'''