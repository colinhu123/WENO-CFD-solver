import numpy as np
import matplotlib.pyplot as plt

def wenoL(u):
    u_con = u[:,:3]
    u0 = u_con[:-5,:]
    u1 = u_con[1:-4,:]
    u2 = u_con[2:-3,:]
    u3 = u_con[3:-2,:]
    u4 = u_con[4:-1,:]

    beta0 = 13/12*(u2-2*u3+u4)**2+1/4*(3*u2-4*u3+u4)**2
    beta1 = 13/12*(u1-2*u2+u3)**2+1/3*(u1-u3)**2
    beta2 = 13/12*(u0-2*u1+u2)**2+1/4*(u0-4*u1+3*u2)**2

    d0 = 0.3
    d1 = 0.6
    d2 = 0.1
    eps = 1e-6
    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    return (w0*(-u4+5*u3+2*u2)+w1*(-u1+5*u2+2*u3)+w2*(2*u0-7*u1+11*u2))/6

def wenoR(u):
    u_con = u[:,:3]
    u0 = u_con[1:-4,:]
    u1 = u_con[2:-3,:]
    u2 = u_con[3:-2,:]
    u3 = u_con[4:-1,:]
    u4 = u_con[5:,:]

    beta0 = 13/12*(u2-2*u3+u4)**2+1/4*(3*u2-4*u3+u4)**2
    beta1 = 13/12*(u1-2*u2+u3)**2+1/3*(u1-u3)**2
    beta2 = 13/12*(u0-2*u1+u2)**2+1/4*(u0-4*u1+3*u2)**2

    d0 = 0.1
    d1 = 0.6
    d2 = 0.3
    eps = 1e-6
    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    return (w0*(2*u4-7*u3+11*u2)+w1*(-u3+5*u2+2*u1)+w2*(-u0+5*u1+2*u2))/6

def flux_x(u,gamma):
    u_con = u[:,:3]
    F = np.zeros_like(u_con)
    u = u_con[:,1]/u_con[:,0]
    p = (gamma-1)*(u_con[:,2]-0.5*u_con[:,0]*u**2)
    F[:,0] = u_con[:,1]
    F[:,1] = u_con[:,0]*u**2+p
    F[:,2] = (u_con[:,2]+p)*u
    return F

def con2primi(u,gamma):
    _eps = 1e-12
    u_con = u[:,:3]
    v = u_con[:,1]/u_con[:,0]
    p = (gamma-1)*(u_con[:,2]-0.5*u_con[:,0]*v**2)
    rho_safe = np.maximum(u_con[:,0],_eps)
    p_safe = np.maximum(p,_eps)
    a = np.sqrt(gamma*p_safe/rho_safe)
    return rho_safe,v,p_safe,a


def HLLC(uL,uR,gamma):
    F_L = flux_x(uL,gamma)
    F_R = flux_x(uR,gamma)

    rhoL,vL,pL,aL = con2primi(uL,gamma)
    rhoR,vR,pR,aR = con2primi(uR,gamma)

    SL = np.minimum(vL - aL, vR - aR)
    SR = np.maximum(vL + aL, vR + aR)

    Sstar = (pR-pL+rhoL*vL*(SL-vL)-rhoR*vR*(SR-vR))/(rhoL*(SL-vL)-rhoR*(SR-vR))

    QstarL = np.ones_like(uL)
    QstarL[:,1] = Sstar
    QstarL[:,2] = uL[:,2]/uL[:,0] + (Sstar-vL)*(Sstar+pL/(rhoL*(SL-vL)))

    QstarL = rhoL[...,None]*(SL[...,None]-vL[...,None])/(SL[...,None]-Sstar[...,None])*QstarL

    QstarR = np.ones_like(uR)
    QstarR[:,1] = Sstar
    QstarR[:,2] = uR[:,2]/uR[:,0] + (Sstar-vR)*(Sstar+pR/(rhoR*(SR-vR)))

    QstarR = rhoR[...,None]*(SR[...,None]-vR[...,None])/(SR[...,None]-Sstar[...,None])*QstarR

    F = np.empty_like(uL)
    
    mask_L = SL >= 0
    if np.any(mask_L):
        F[mask_L] = F_L[mask_L]

    mask_Ls = (SL < 0) & (Sstar >= 0)
    if np.any(mask_Ls):
        F[mask_Ls] = F_L[mask_Ls] + SL[mask_Ls][...,None] * (QstarL[mask_Ls] - uL[mask_Ls])

    mask_sR = (Sstar < 0) & (SR > 0)
    if np.any(mask_sR):
        F[mask_sR] = F_R[mask_sR] + SR[mask_sR][...,None] * (QstarR[mask_sR] - uR[mask_sR])

    mask_R = SR <= 0
    if np.any(mask_R):
        F[mask_R] = F_R[mask_R]

    return F

def L(u,gamma,dx):
    uL = wenoL(u)
    uR = wenoR(u)

    F = HLLC(uL,uR,gamma)

    return -(F[1:,:]-F[:-1,:])/dx



def RK3(u,gamma,dx,dt):
    u1 = u.copy()
    u2 = u.copy()

    u1[3:-3, :3] = u[3:-3,:3] + dt*L(u,gamma,dx)
    u2[3:-3,:3] = 0.75*u[3:-3,:3]+0.25*u1[3:-3,:3]+dt*L(u1,gamma,dx)
    u[3:-3,:3] = 1/3*u[3:-3,:3]+2/3*u2[3:-3,:3]+2/3*dt*L(u2,gamma,dx)


    return u

def main(Ngrid, Nstep):
    CFL = 0.1
    u = np.zeros((Ngrid+6,6))
    gamma = 1.4

    dx = 1/Ngrid

    half_index = int((Ngrid+6)/2)
    u[:half_index,3] = 1.0
    u[half_index:,3] = 0.125
    u[:half_index,5] = 1.0
    u[half_index:,5] = 0.1
    u[:,4] = 0.0

    u[:,0] = u[:,3]
    u[:,1] = u[:,3]*u[:,4]
    u[:,2] = 0.5*u[:,3]*u[:,4]**2+u[:,5]/(gamma-1)



    for i in range(Nstep):
        rho,v,p,a = con2primi(u,gamma)
        dt = dx*CFL/np.max(a+np.abs(v))
        u = RK3(u,gamma,dx,dt)

    rho,v,p,a = con2primi(u,gamma)
    u[:,3] = rho
    u[:,4] = v
    u[:,5] = p
    return u

u = main(1000,400)

plt.plot(np.arange(len(u[:,3])),u[:,3])
plt.show()