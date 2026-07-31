
import time
import matplotlib.pyplot as plt
import numpy as np
import weno_ext

grid_dict = {
    "nx": 40,
    "ny": 10,
    "x" : 4,
    "y": 1,
    "dx": 1/100
}

control_dict = {
    "nstep": 100,
    "CFL": 0.1,
    "min_step_time": 1e-10,
    "max_time_step": 0.1,
    "file_storage":False,
    "force_hlle": False,
    "jp_cri":(1,5)
}

phys_dict = {
    "gamma":1.4
}



import numpy as np

def init(grid_dict, control_dict=None, gamma=1.4):
    """
    Initialize primitive fields (rho, u, v, p) on a grid with NG=3 ghost cells,
    apply boundary conditions, then convert to conserved variables and return q
    with shape (nx+6, ny+6, 8) where:
      - q[..., 0:4] are conserved: [rho, rho*u, rho*v, E]
      - q[..., 4:8] are primitive: [rho, u, v, p]
    Assumptions:
      - grid_dict contains "nx", "ny", "x" (domain length in x), "y" (domain length in y)
      - NG (ghost cells) = 3
    """
    nx = grid_dict["nx"]
    ny = grid_dict["ny"]
    Lx = grid_dict["x"]
    Ly = grid_dict["y"]
    NG = 3

    # allocate q: conserved (0..3) + primitive (4..7)
    q = np.zeros((nx + 2*NG, ny + 2*NG, 8), dtype=float)

    # primitive indices in q[...,4:8]
    # default ambient (pre-shock) state
    ambient = [1.4, 0.0, 0.0, 1.0]   # [rho, u, v, p]
    post_shock = [8.0, 4.125, -7.1447, 116.5]

    # fill whole domain initially with ambient
    q[:, :, 4] = ambient[0]   # rho
    q[:, :, 5] = ambient[1]   # u
    q[:, :, 6] = ambient[2]   # v
    q[:, :, 7] = ambient[3]   # p

    # shock line function (y as function of x)
    def shock(x):
        return np.sqrt(3.0) * x - np.sqrt(3.0) / 6.0

    # fill interior based on shock geometry:
    # interior i indices run from NG .. NG+nx-1
    for i in range(NG, NG + nx):
        # physical x coordinate at cell center (approx): map i -> [0, Lx]
        x_pos = (i - NG + 0.5) / nx * Lx
        y_shock = shock(x_pos)

        # compute j index corresponding to shock y: interior j from NG..NG+ny-1
        j_shock = int(np.floor((y_shock / Ly) * ny)) + NG

        # clamp to interior range
        j_shock = max(NG, min(NG + ny, j_shock))

        # set post-shock primitive state for cells with y >= shock (i.e., j >= j_shock)
        if j_shock < NG + ny:
            q[i, j_shock: NG + ny, 4] = post_shock[0]
            q[i, j_shock: NG + ny, 5] = post_shock[1]
            q[i, j_shock: NG + ny, 6] = post_shock[2]
            q[i, j_shock: NG + ny, 7] = post_shock[3]

    # now convert primitives to conserved variables for the whole array
    rho = q[:, :, 4]
    u = q[:, :, 5]
    v = q[:, :, 6]
    p = q[:, :, 7]

    q[:, :, 0] = rho
    q[:, :, 1] = rho * u
    q[:, :, 2] = rho * v
    q[:, :, 3] = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    # apply boundary conditions on the primitive fields, then refresh conserved
    q = apply_bc(q, grid_dict, NG=NG, post_shock=post_shock)

    # recompute conserved (in case BCs modified primitives)
    rho = q[:, :, 4]
    u = q[:, :, 5]
    v = q[:, :, 6]
    p = q[:, :, 7]
    q[:, :, 0] = rho
    q[:, :, 1] = rho * u
    q[:, :, 2] = rho * v
    q[:, :, 3] = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)

    return q

def apply_bc(u, grid_info, NG=3, post_shock=None):
    """
    Apply boundary conditions to array `u` (which stores primitives in u[...,4:8]).
    - Left ghost cells (i < NG) are set to post_shock primitive state (inflow/post-shock)
    - Right ghost cells (i >= NG+nx) copy the last interior column (outflow)
    - Bottom ghosts (j < NG): reflective (mirror) with v-component reversed
    - Top ghosts (j >= NG+ny): copy last interior row (outflow/top)
    NOTE: u shape is (nx+2*NG, ny+2*NG, 8)
    """
    nx = grid_info["nx"]
    ny = grid_info["ny"]
    Lx = grid_info["x"]
    Ly = grid_info["y"]

    if post_shock is None:
        post_shock = [8.0, 4.125, -7.1447, 116.5]

    # indices for interior
    i0 = NG
    i1 = NG + nx - 1
    j0 = NG
    j1 = NG + ny - 1

    # ---- Left boundary: set ghost cells to post_shock primitive state ----
    for i in range(0, NG):
        u[i, :, 4] = post_shock[0]
        u[i, :, 5] = post_shock[1]
        u[i, :, 6] = post_shock[2]
        u[i, :, 7] = post_shock[3]

    # ---- Right boundary: copy last interior column into right ghosts (outflow) ----
    for i in range(i1 + 1, NG + nx + NG):
        # copy from i1
        u[i, :, :] = u[i1, :, :]

    # ---- Bottom boundary (reflective): mirror across j=NG line, reverse v ----
    bound = int(1/6*nx/Lx+2)
    for j in range(0, NG):
        jm = 2*NG - j - 1  # mirror index inside domain
        # copy entire row from jm then flip v (index 6) sign
        u[bound:, j, :] = u[bound:, jm, :]
        u[bound:, j, 6] = -u[bound:, jm, 6]
        u[bound:, j, 2] = -u[bound:, jm, 2]

        u[:bound, j, 4] = post_shock[0]
        u[:bound, j, 5] = post_shock[1]
        u[:bound, j, 6] = post_shock[2]
        u[:bound, j, 7] = post_shock[3]

    # ---- Top boundary: copy last interior row into top ghosts (outflow) ----
    for j in range(j1 + 1, NG + ny + NG):
        u[:, j, :] = u[:, j1, :]

    return u

def init_refac(grid_dict,control_dict,gamma=1.4):
    nx = grid_dict["nx"]
    ny = grid_dict["ny"]
    lx = grid_dict["x"]
    ly = grid_dict["y"]
    dx = grid_dict["dx"]

    q = np.zeros((nx+6,ny+6,8))

    q[:,:,4] = 1.4
    q[:,:,5] = 0
    q[:,:,6] = 0
    q[:,:,7] = 1

    def shock(x):
        return np.sqrt(3)*x - np.sqrt(3)/6
    bound = 0
    for i in range(3,nx+3):
        x = (i-3+0.5)/nx*lx
        y_pos = shock(x)
        j_phy_domain = int(np.round(y_pos/ly*ny))
        #print(j_phy_domain)
        if j_phy_domain == 0:
            bound = i
        j_phy_domain = np.maximum(j_phy_domain,0)        
        q[i,j_phy_domain+3:,4] = 8
        q[i,j_phy_domain+3:,5] = 4.125
        q[i,j_phy_domain+3:,6] = -7.1447
        q[i,j_phy_domain+3:,7] = 116.5

    q[3:nx+3, 3:ny+3, 0] = q[3:nx+3, 3:ny+3, 4]
    q[3:nx+3, 3:ny+3, 1] = q[3:nx+3, 3:ny+3, 4]* q[3:nx+3, 3:ny+3, 5]
    q[3:nx+3, 3:ny+3, 2] = q[3:nx+3, 3:ny+3, 4]* q[3:nx+3, 3:ny+3, 6]
    q[3:nx+3, 3:ny+3, 3] = q[3:nx+3, 3:ny+3, 7]/(gamma-1) + \
                            0.5* q[3:nx+3, 3:ny+3,4]*(q[3:nx+3, 3:ny+3,5]**2 + q[3:nx+3, 3:ny+3,6]**2)

    return apply_bc_refac(q,bound,grid_dict),bound

def apply_bc_refac(q,bound,grid_dict):
    nx = grid_dict["nx"]
    ny = grid_dict["ny"]
    lx = grid_dict["x"]
    ly = grid_dict["y"]
    dx = grid_dict["dx"]

    for i in range(3):
        q[i,:,:] = [8,33,-57.1576,563.499,8,4.125,-7.1447,116.5]
        q[:,ny+3+i,:] = q[:,ny+2,:]
        q[nx+3+i,:,:] = q[nx+2,:,:]

        q[:bound+1,i,:] = [8,33,-57.1576,563.499,8,4.125,-7.1447,116.5]

        q[bound+1:bound+10,i,:] = q[bound+1:bound+10,3,:]

        q[bound+10:,i,:] = q[bound+10:,5-i,:]
        q[bound+10:, i, 2] *= -1
        q[bound+10:, i, 6] *= -1

    return q

if __name__ == "__main__":
    q1,bound = init_refac(grid_dict, control_dict,gamma = 1.4)
    print(bound)
    ql,fl,qr,fr = weno_ext.weno_y_py(q1,1.4)
    fluxy = weno_ext.hllc_y_rs_py(ql,qr,fl,fr,1.4)
    plt.imshow(q1[:,:,2].T)
    plt.show()