import matplotlib.pyplot as plt
import numpy as np
import time
import os
import datetime

eps = 1e-12
d00 = (210-3**0.5)/1080
d01 = 11/18
d02 = (210+3**0.5)/1080

SQR3_12 = 3**0.5/12


gamma = 1.4



def weno_L_1x(u):
    '''
    :param: u shape is Ngrid+4,Ngrid+4,8

    return matrix is Ngrid+1,Ngrid+1,4

    This reconstruction procedure is for the first weno reconstruction on 2d plane
    '''
    u_con = u[:,:,:4]
    u0 = u_con[:-5,:,:]
    u1 = u_con[1:-4,:,:]
    u2 = u_con[2:-3,:,:]
    u3 = u_con[3:-2,:,:]
    u4 = u_con[4:-1,:,:]
    beta0 = 13/12*(u2-2*u3+u4)**2+1/4*(3*u2-4*u3+u4)**2
    beta1 = 13/12*(u1-2*u2+u3)**2+1/4*(u1-u3)**2
    beta2 = 13/12*(u0-2*u1+u2)**2+1/4*(u0-4*u1+3*u2)**2

    d0 = 0.3
    d1 = 0.6
    d2 = 0.1
    alpha0 = d0/(eps+beta0)**2
    alpha1 = d1/(eps+beta1)**2
    alpha2 = d2/(eps+beta2)**2
    sum_alpha = alpha0+alpha1+alpha2
    w0 = alpha0/sum_alpha
    w1 = alpha1/sum_alpha
    w2 = alpha2/sum_alpha
    uL = (w0*(-u4+5*u3+2*u2)+w1*(-u1+5*u2+2*u3)+w2*(2*u0-7*u1+11*u2))/6
    return uL

def weno_R_1x(u):
    '''
    u: Ngrid+6 * Ngrid+6 * 8

    return matrix with size Ngrid+1* Ngrid+6*4

    reconstruction right state of field on x-direction
    '''
    u_con = u[:,:,:4]
    u0 = u_con[1:-4,:,:]
    u1 = u_con[2:-3,:,:]
    u2 = u_con[3:-2,:,:]
    u3 = u_con[4:-1,:,:]
    u4 = u_con[5:,:,:]

    beta0 = 13/12*(u2-2*u3+u4)**2+1/4*(3*u2-4*u3+u4)**2
    beta1 = 13/12*(u1-2*u2+u3)**2+1/4*(u1-u3)**2
    beta2 = 13/12*(u0-2*u1+u2)**2+1/4*(u0-4*u1+3*u2)**2

    d0 = 0.1
    d1 = 0.6
    d2 = 0.3
    alpha0 = d0/(eps+beta0)**2
    alpha1 = d1/(eps+beta1)**2
    alpha2 = d2/(eps+beta2)**2
    sum_alpha = alpha0+alpha1+alpha2
    w0 = alpha0/sum_alpha
    w1 = alpha1/sum_alpha
    w2 = alpha2/sum_alpha
    uR = (w0*(2*u4-7*u3+11*u2)+w1*(-u3+5*u2+2*u1)+w2*(-u0+5*u1+2*u2))/6
    return uR

def weno_2_y(u,Ngrid):
    '''
    This function is to the second step to reconstruct along y direction, supposed to call after `weno_L/R_x` 
    u: Ngrid+1 * Ngrid+6 * 4

    return matrix shape is Ngrid+1*Ngrid*4
    '''
    u_con = u[:,1:-1,:4]
    u0 = u_con[:,:-4,:]
    u1 = u_con[:,1:-3,:]
    u2 = u_con[:,2:-2,:]
    u3 = u_con[:,3:-1,:]
    u4 = u_con[:,4:,:]
    beta0 = 13/12*(u2-2*u3+u4)**2+1/4*(3*u2-4*u3+u4)**2
    beta1 = 13/12*(u1-2*u2+u3)**2+1/4*(u1-u3)**2
    beta2 = 13/12*(u0-2*u1+u2)**2+1/4*(u0-4*u1+3*u2)**2

    alpha0 = d00/(eps+beta0)**2
    alpha1 = d01/(eps+beta1)**2
    alpha2 = d02/(eps+beta2)**2
    sum_alpha = alpha0+alpha1+alpha2
    w0 = alpha0/sum_alpha
    w1 = alpha1/sum_alpha
    w2 = alpha2/sum_alpha
    uq1 = w0*(u2+SQR3_12*(3*u2-4*u3+u4))+w1*(u2-(-u1+u3)*SQR3_12)+w2*(u2-(3*u2-4*u1+u0)*SQR3_12)
    flux1 = flux_x(uq1,Ngrid)

    alpha0 = d02/(eps+beta0)**2
    alpha1 = d01/(eps+beta1)**2
    alpha2 = d00/(eps+beta2)**2
    sum_alpha = alpha0+alpha1+alpha2
    w0 = alpha0/sum_alpha
    w1 = alpha1/sum_alpha
    w2 = alpha2/sum_alpha
    uq2 = w0*(u2-SQR3_12*(3*u2-4*u3+u4))+w1*(u2+(-u1+u3)*SQR3_12)+w2*(u2+(3*u2-4*u1+u0)*SQR3_12)
    flux2 = flux_x(uq2,Ngrid)

    return 0.5*(flux1+flux2),0.5*(uq1+uq2)

def weno_L_1y(u):
    u_con = u[:,:,:4]
    u0 = u_con[:,:-5,:]
    u1 = u_con[:,1:-4,:]
    u2 = u_con[:,2:-3,:]
    u3 = u_con[:,3:-2,:]
    u4 = u_con[:,4:-1,:]

    beta0 = 13/12*(u2-2*u3+u4)**2+1/4*(3*u2-4*u3+u4)**2
    beta1 = 13/12*(u1-2*u2+u3)**2+1/4*(u1-u3)**2
    beta2 = 13/12*(u0-2*u1+u2)**2+1/4*(u0-4*u1+3*u2)**2

    d0 = 0.3
    d1 = 0.6
    d2 = 0.1
    alpha0 = d0/(eps+beta0)**2
    alpha1 = d1/(eps+beta1)**2
    alpha2 = d2/(eps+beta2)**2
    sum_alpha = alpha0+alpha1+alpha2
    w0 = alpha0/sum_alpha
    w1 = alpha1/sum_alpha
    w2 = alpha2/sum_alpha
    uL = (w0*(-u4+5*u3+2*u2)+w1*(-u1+5*u2+2*u3)+w2*(2*u0-7*u1+11*u2))/6
    return uL

def weno_R_1y(u):
    u_con = u[:,:,:4]
    u0 = u_con[:,1:-4,:]
    u1 = u_con[:,2:-3,:]
    u2 = u_con[:,3:-2,:]
    u3 = u_con[:,4:-1,:]
    u4 = u_con[:,5:,:]

    beta0 = 13/12*(u2-2*u3+u4)**2+1/4*(3*u2-4*u3+u4)**2
    beta1 = 13/12*(u1-2*u2+u3)**2+1/4*(u1-u3)**2
    beta2 = 13/12*(u0-2*u1+u2)**2+1/4*(u0-4*u1+3*u2)**2

    d0 = 0.3
    d1 = 0.6
    d2 = 0.1
    alpha0 = d0/(eps+beta0)**2
    alpha1 = d1/(eps+beta1)**2
    alpha2 = d2/(eps+beta2)**2
    sum_alpha = alpha0+alpha1+alpha2
    w0 = alpha0/sum_alpha
    w1 = alpha1/sum_alpha
    w2 = alpha2/sum_alpha
    uR = (w0*(2*u4-7*u3+11*u2)+w1*(-u3+5*u2+2*u1)+w2*(-u0+5*u1+2*u2))/6
    return uR

def weno_2_x(u,Ngrid):
    '''
    This function is to the second step to reconstruct along y direction, supposed to call after `weno_L/R_x` 
    u: Ngrid+6 * Ngrid+6 * 4
    return matix shape is Ngrid*Ngrid+1*4
    '''
    u_con = u[1:-1,:,:4]
    u0 = u_con[:-4,:,:]
    u1 = u_con[1:-3,:,:]
    u2 = u_con[2:-2,:,:]
    u3 = u_con[3:-1,:,:]
    u4 = u_con[4:,:,:]
    beta0 = 13/12*(u2-2*u3+u4)**2+1/4*(3*u2-4*u3+u4)**2
    beta1 = 13/12*(u1-2*u2+u3)**2+1/4*(u1-u3)**2
    beta2 = 13/12*(u0-2*u1+u2)**2+1/4*(u0-4*u1+3*u2)**2

    alpha0 = d00/(eps+beta0)**2
    alpha1 = d01/(eps+beta1)**2
    alpha2 = d02/(eps+beta2)**2
    sum_alpha = alpha0+alpha1+alpha2
    w0 = alpha0/sum_alpha
    w1 = alpha1/sum_alpha
    w2 = alpha2/sum_alpha
    uq1 = w0*(u2+SQR3_12*(3*u2-4*u3+u4))+w1*(u2-(-u1+u3)*SQR3_12)+w2*(u2-(3*u2-4*u1+u0)*SQR3_12)
    flux1 = flux_y(uq1,Ngrid)

    alpha0 = d02/(eps+beta0)**2
    alpha1 = d01/(eps+beta1)**2
    alpha2 = d00/(eps+beta2)**2
    sum_alpha = alpha0+alpha1+alpha2
    w0 = alpha0/sum_alpha
    w1 = alpha1/sum_alpha
    w2 = alpha2/sum_alpha
    uq2 = w0*(u2-SQR3_12*(3*u2-4*u3+u4))+w1*(u2+(-u1+u3)*SQR3_12)+w2*(u2+(3*u2-4*u1+u0)*SQR3_12)
    flux2 = flux_y(uq2,Ngrid)

    return 0.5*(flux1+flux2),0.5*(uq1+uq2)

import numpy as np

_eps = 1e-12

def conserved_to_prim(U, gamma):
    """U: (...,4) conserved -> returns (rho, u, v, p, a) arrays"""
    rho = U[...,0]
    rho_safe = np.maximum(rho, _eps)
    u = U[...,1] / rho_safe
    v = U[...,2] / rho_safe
    E = U[...,3]
    kinetic = 0.5 * rho_safe * (u**2 + v**2)
    e_int = E - kinetic
    p = (gamma - 1.0) * e_int
    # avoid negative pressure in intermediate calc (just for sound speed); caller should handle negativity elsewhere
    p_safe = np.maximum(p, _eps)
    a = np.sqrt(gamma * p_safe / rho_safe)
    return rho, u, v, p, a

def flux_x(U, gamma):
    """Physical flux F_x(U) for conserved U (...,4)."""
    rho, u, v, p, a = conserved_to_prim(U, gamma)
    F = np.empty_like(U)
    F[...,0] = rho * u
    F[...,1] = rho * u**2 + p
    F[...,2] = rho * u * v
    F[...,3] = (U[...,3] + p) * u
    return F

def flux_y(U, gamma):
    """Physical flux F_y(U) for conserved U (...,4)."""
    rho, u, v, p, a = conserved_to_prim(U, gamma)
    G = np.empty_like(U)
    G[...,0] = rho * v
    G[...,1] = rho * u * v
    G[...,2] = rho * v**2 + p
    G[...,3] = (U[...,3] + p) * v
    return G

def _hllc_common(UL, UR, gamma, direction='x'):
    """
    Core HLLC routine. UL,UR: (...,4) conserved states on left/right of interface.
    direction: 'x' or 'y' - determines which velocity is normal (u for x, v for y).
    Returns flux (...,4).
    """
    # primitives
    rhoL, uL_x, uL_y, pL, aL = conserved_to_prim(UL, gamma)
    rhoR, uR_x, uR_y, pR, aR = conserved_to_prim(UR, gamma)

    if direction == 'x':
        unL = uL_x; utL = uL_y
        unR = uR_x; utR = uR_y
        F_L = flux_x(UL, gamma)
        F_R = flux_x(UR, gamma)
    else:
        unL = uL_y; utL = uL_x
        unR = uR_y; utR = uR_x
        F_L = flux_y(UL, gamma)
        F_R = flux_y(UR, gamma)

    # estimate wave speeds (Davis estimate)
    SL = np.minimum(unL - aL, unR - aR)
    SR = np.maximum(unL + aL, unR + aR)

    # avoid exact zero width
    SL = np.minimum(SL, -_eps)
    SR = np.maximum(SR, _eps)

    # middle wave speed S_star (Toro)
    # numerator = pR - pL + rhoL*unL*(SL - unL) - rhoR*unR*(SR - unR)
    num = (pR - pL) + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR)
    den = rhoL * (SL - unL) - rhoR * (SR - unR)
    # protect divide by zero
    den_safe = np.where(np.abs(den) < _eps, np.sign(den) * _eps + _eps, den)
    Sstar = num / den_safe

    # compute star-region conserved states U*_L and U*_R
    # density in star
    rho_star_L = rhoL * (SL - unL) / (SL - Sstar)
    rho_star_R = rhoR * (SR - unR) / (SR - Sstar)

    # pressure in star (from Rankine-Hugoniot)
    p_star_L = pL + rhoL * (SL - unL) * (Sstar - unL)
    p_star_R = pR + rhoR * (SR - unR) * (Sstar - unR)
    # numeric safeguard: average if small inconsistency
    p_star = 0.5 * (p_star_L + p_star_R)

    # energy in star: use conservation form
    EL = UL[...,3]; ER = UR[...,3]
    E_star_L = ((SL - unL) * EL - pL * unL + p_star * Sstar) / (SL - Sstar)
    E_star_R = ((SR - unR) * ER - pR * unR + p_star * Sstar) / (SR - Sstar)

    # assemble Ustar arrays (...,4)
    U_star_L = np.empty_like(UL)
    U_star_R = np.empty_like(UR)

    # normal momentum = rho_star * Sstar ; tangential momentum preserved: rho_star * utL
    U_star_L[...,0] = rho_star_L
    U_star_L[...,1] = rho_star_L * Sstar if direction == 'x' else rho_star_L * utL
    U_star_L[...,2] = rho_star_L * utL if direction == 'x' else rho_star_L * Sstar
    U_star_L[...,3] = E_star_L

    U_star_R[...,0] = rho_star_R
    U_star_R[...,1] = rho_star_R * Sstar if direction == 'x' else rho_star_R * utR
    U_star_R[...,2] = rho_star_R * utR if direction == 'x' else rho_star_R * Sstar
    U_star_R[...,3] = E_star_R

    # HLLC flux selection
    # Cases:
    # SL >= 0 : flux = F_L
    # SL <= 0 <= Sstar : flux = F_L + SL*(U_star_L - UL)
    # Sstar <= 0 <= SR : flux = F_R + SR*(U_star_R - UR)
    # SR <= 0 : flux = F_R

    # Start with zeros
    F = np.empty_like(UL)

    # region where SL >= 0
    mask_L = SL >= 0
    if np.any(mask_L):
        F[mask_L] = F_L[mask_L]

    # region where SL < 0 and Sstar >= 0 (i.e., SL < 0 <= Sstar)
    mask_Ls = (SL < 0) & (Sstar >= 0)
    if np.any(mask_Ls):
        F[mask_Ls] = F_L[mask_Ls] + SL[mask_Ls][...,None] * (U_star_L[mask_Ls] - UL[mask_Ls])

    # region where Sstar < 0 and SR > 0 (i.e., Sstar < 0 <= SR)
    mask_sR = (Sstar < 0) & (SR > 0)
    if np.any(mask_sR):
        F[mask_sR] = F_R[mask_sR] + SR[mask_sR][...,None] * (U_star_R[mask_sR] - UR[mask_sR])

    # region where SR <= 0
    mask_R = SR <= 0
    if np.any(mask_R):
        F[mask_R] = F_R[mask_R]

    return F

def hllc_flux_x(UL, UR, gamma):
    """HLLC flux in x-direction at interfaces between UL (left) and UR (right)."""
    return _hllc_common(UL, UR, gamma, direction='x')

def hllc_flux_y(UL, UR, gamma):
    """HLLC flux in y-direction at interfaces between UL (left) and UR (right)."""
    return _hllc_common(UL, UR, gamma, direction='y')

def L(u,Ngrid):
    fluxLx,qLx = weno_2_y(weno_L_1x(u),Ngrid)
    fluxRx,qRx = weno_2_y(weno_R_1x(u),Ngrid)
    fluxLy,qLy = weno_2_x(weno_L_1y(u),Ngrid)
    fluxRy,qRy = weno_2_x(weno_R_1y(u),Ngrid)
    flux_x = hllc_flux_x(qLx,qRx,1.4)
    flux_y = hllc_flux_y(qLy,qRy,1.4)
    dx = 1/Ngrid

    df_dx = (flux_x[1:,:,:]-flux_x[:-1,:,:])/dx
    dg_dy = (flux_y[:,1:,:]-flux_y[:,:-1,:])/dx

    return -(df_dx+dg_dy)

def apply_bc(u, Ngrid):
    '''
    直接在原数组上应用边界条件
    '''
    # 行边界
    u[0, :, :] = u[3, :, :]
    u[1, :, :] = u[3, :, :] 
    u[2, :, :] = u[3, :, :]
    u[Ngrid+3, :, :] = u[Ngrid+2, :, :]
    u[Ngrid+4, :, :] = u[Ngrid+2, :, :]
    u[Ngrid+5, :, :] = u[Ngrid+2, :, :]
    
    # 列边界
    u[:, 0, :] = u[:, 3, :]
    u[:, 1, :] = u[:, 3, :]
    u[:, 2, :] = u[:, 3, :]
    u[:, Ngrid+3, :] = u[:, Ngrid+2, :]
    u[:, Ngrid+4, :] = u[:, Ngrid+2, :]
    u[:, Ngrid+5, :] = u[:, Ngrid+2, :]
    
    return u

def apply_bc_corrected(u, Ngrid):
    '''
    正确的边界条件：只使用内部网格边界值填充幽灵区域
    '''
    # 内部网格索引范围
    i_start, i_end = 3, 3 + Ngrid  # 3 到 12 (不包含12)
    j_start, j_end = 3, 3 + Ngrid
    
    # 左边界幽灵行 (0,1,2) 用内部网格左边界 (第3行) 填充
    for i in [0, 1, 2]:
        u[i, :, :] = u[3, :, :]  # 用第3行所有列填充
    
    # 右边界幽灵行 (Ngrid+3, Ngrid+4, Ngrid+5) 用内部网格右边界 (第Ngrid+2行) 填充
    for i in [Ngrid+3, Ngrid+4, Ngrid+5]:
        u[i, :, :] = u[Ngrid+2, :, :]
    
    # 下边界幽灵列 (0,1,2) 用内部网格下边界 (第3列) 填充
    for j in [0, 1, 2]:
        u[:, j, :] = u[:, 3, :]
    
    # 上边界幽灵列 (Ngrid+3, Ngrid+4, Ngrid+5) 用内部网格上边界 (第Ngrid+2列) 填充
    for j in [Ngrid+3, Ngrid+4, Ngrid+5]:
        u[:, j, :] = u[:, Ngrid+2, :]
    
    return u

def RK3(u,Ngrid,dt):
    u1 = u.copy()
    u2 = u.copy()
    u1 = apply_bc_corrected(u1,Ngrid)
    u2[3:Ngrid+3,3:Ngrid+3,:4] = 0.75*u[3:Ngrid+3,3:Ngrid+3,:4]+0.25*u1[3:Ngrid+3,3:Ngrid+3,:4]+dt*L(u1,Ngrid)
    u2 = apply_bc_corrected(u2,Ngrid)
    u[3:Ngrid+3,3:Ngrid+3,:4] = 1/3*u[3:Ngrid+3,3:Ngrid+3,:4] + 2/3*u2[3:Ngrid+3,3:Ngrid+3,:4] + 2/3*dt*L(u2,Ngrid)
    u = apply_bc_corrected(u,Ngrid)
    return u

def con_var2primi(u):
    u1 = u.copy()
    rho = u1[:,:,0]
    rho = np.maximum(rho,1e-10)
    u1[:,:,4] = rho
    u1[:,:,5] = u1[:,:,1]/rho
    u1[:,:,6] = u1[:,:,2]/rho
    u1[:,:,7] = (gamma-1)*(u1[:,:,3]-0.5*rho*(u1[:,:,5]**2+u1[:,:,6]**2))

    return u1
def con_var2primi_pfloor(u):
    # ... your existing code to compute rho,uvel,vvel,E ...
    rho = np.maximum(u[:,:,0], 1e-12)
    uvel = u[:,:,1]/rho
    vvel = u[:,:,2]/rho
    E = u[:,:,3]
    kinetic = 0.5 * rho * (uvel**2 + vvel**2)
    e_internal = E - kinetic
    p = (gamma-1) * e_internal

    # pressure floor
    pmin = 1e-8   # tune: 1e-8 .. 1e-6 depending on problem scale
    mask = p < pmin
    if np.any(mask):
        e_internal[mask] = pmin / (gamma - 1)
        E[mask] = kinetic[mask] + e_internal[mask]
        p[mask] = pmin
        u[:,:,3] = E  # update conserved energy to remain consistent

    # fill primitive fields (indices you use)
    u[:,:,4] = rho
    u[:,:,5] = uvel
    u[:,:,6] = vvel
    u[:,:,7] = p
    return u

def main(Ngrid,N_STEP):
    u = np.ones((Ngrid+6,Ngrid+6,8))
    current_time = datetime.datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(folder_name,exist_ok=True)

    dx = 1/Ngrid
    #N_STEP = 50
    u = np.zeros((Ngrid+6,Ngrid+6,8))
    CFL = 0.2

    quad_index = int((Ngrid+6)/2)

    u[:quad_index,:quad_index,4] = 0.125
    u[:quad_index,:quad_index,5] = 1
    u[:quad_index,:quad_index,6] = 1
    u[:quad_index,:quad_index,7] = 0.125

    u[quad_index:,:quad_index,4] = 0.5
    u[quad_index:,:quad_index,5] = 0
    u[quad_index:,:quad_index,6] = 1
    u[quad_index:,:quad_index,7] = 0.25

    u[:quad_index,quad_index:,4] = 0.5
    u[:quad_index,quad_index:,5] = 1
    u[:quad_index,quad_index:,6] = 0
    u[:quad_index,quad_index:,7] = 0.25

    u[quad_index:,quad_index:,4] = 1
    u[quad_index:,quad_index:,5] = 0
    u[quad_index:,quad_index:,6] = 0
    u[quad_index:,quad_index:,7] = 1

    u[:, :, 0] = u[:, :, 4]
    # rho*u -> momentum_x
    u[:, :, 1] = u[:, :, 4] * u[:, :, 5]
    # rho*v -> momentum_y
    u[:, :, 2] = u[:, :, 4] * u[:, :, 6]
    # p/(gamma-1) + 0.5*rho*v^2 -> Total Energy E
    u[:, :, 3] = u[:, :, 7] / (gamma - 1) + 0.5 * u[:, :, 4] * (u[:, :, 5]**2 + u[:, :, 6]**2)

    u = apply_bc_corrected(u,Ngrid)
    
    print(u[:,:,4])
    np.save(os.path.join(folder_name,"0.npy"),u)
    t_list = np.array([0])
    T = 0
    for i in range(1,N_STEP):
        print(i)
        c_max = np.max(np.sqrt(gamma*u[:,:,7]/np.maximum(u[:,:,0],1e-10)))

        dt =  CFL*dx/c_max

        u = RK3(u,Ngrid,dt)

        u = con_var2primi(u)

        np.save(os.path.join(folder_name,f"{i}.npy"),u)
        T += dt
        t_list = np.append(t_list,T)

    np.save(os.path.join(folder_name,"time.npy"),t_list)



main(2048,100)

