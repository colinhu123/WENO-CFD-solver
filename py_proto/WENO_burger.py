import numpy as np
import math
import matplotlib.pyplot as plt

Ngrid = 100
eps = 1e-6
dx = 1/Ngrid
u = np.ones(Ngrid+6)#include ghoast cell in this case
# the exact domain is from 2 to Ngrid+2

A_minus = np.array([
    [1/3,-7/6,11/6],
    [-1/6,5/6,1/3],
    [1/3,5/6,-1/6]
])

A_plus = np.array(
    [
        [-1/6,5/6,1/3],
        [1/3,5/6,-1/6],
        [11/6,-7/6,1/3]
    ]
)

print(np.shape(u[2:Ngrid+2]))

def weno_reconstruct_minus(stencil):
    '''
    Reconstruct the value at i + 1/2 by applying the stencil from I_{i-2} to I_{i+2} 
    '''
    u1 = np.dot(A_minus[0],stencil[0:3])
    u2 = np.dot(A_minus[1],stencil[1:4])
    u3 = np.dot(A_minus[2],stencil[2:])
    u = np.array([u1,u2,u3])
    beta = weno_smoothness(stencil)
    
    gamma_j = np.array([0.1,0.6,0.3])

    weight_j = gamma_j/(eps+beta)**2
    
    weight = weight_j/weight_j.sum()

    return np.dot(u,weight)

def weno_reconstruct_plus(stencil):
    '''
    Reconstruct by stencil from I_{i-1} to I_{i+3}
    '''
    u1 = np.dot(A_plus[0],stencil[0:3])
    u2 = np.dot(A_plus[1],stencil[1:4])
    u3 = np.dot(A_plus[2],stencil[2:])
    u = np.array([u1,u2,u3])
    beta = weno_smoothness(stencil)
    gamma_j = np.array([0.3,0.6,0.1])

    weight_j = gamma_j/(eps+beta)**2
    
    weight = weight_j/weight_j.sum()

    return np.dot(u,weight)

def weno_smoothness(stencil):
    beta = np.zeros(3)

    for i in range(3):
        b = 13/12*(stencil[i]-2*stencil[i+1]+stencil[i+2])**2+1/4*(stencil[i]-4*stencil[i+1]+3*stencil[i+2])**2
        beta[i] = b

    return beta

def Godunov_flux(u_plus,u_minus):

    def flux(k):
        return k**2/2

    a = min(u_plus,u_minus)
    b = max(u_plus,u_minus)
    candidate = [flux(a),flux(b)]

    if a <= 0.0 <= b:
        candidate.append(flux(0.0))   # equals 0

    if u_minus < u_plus:
        return min(candidate)
    else:
        return max(candidate)
    

def roe_flux_burgers(uL, uR):
    """
    Roe flux for scalar Burgers f(u) = 0.5*u^2.
    uL, uR may be scalars or numpy arrays of the same shape.
    Returns flux across interface (flux from left side).
    """
    fL = 0.5 * uL**2
    fR = 0.5 * uR**2
    a_tilde = 0.5 * (uL + uR)          # Roe average speed
    flux = 0.5 * (fL + fR) - 0.5 * np.abs(a_tilde) * (uR - uL)
    return flux

def L(u):
    u_plus = np.zeros(Ngrid+1)
    u_minus = np.zeros(Ngrid+1)
    for j in range(Ngrid+1):
        u_plus[j] = weno_reconstruct_plus(u[j+1:j+6])
        u_minus[j] = weno_reconstruct_minus(u[j:j+5])
    tmp = np.zeros(Ngrid)
    for i  in range(len(tmp)):
        #tmp[i] = -1/dx*(Godunov_flux(u_minus[i],u_plus[i])-Godunov_flux(u_minus[i+1],u_plus[i+1]))
        tmp[i] = -1/dx*(roe_flux_burgers(u_plus[i],u_minus[i])-
                        roe_flux_burgers(u_plus[i+1],u_minus[i+1]))
    return tmp

def main():
    u = np.zeros(Ngrid+6)
    u[:int((Ngrid+6)/2)] = 0
    u[int((Ngrid+6)/2):] = 1
    Nstep = 40

    CFL = 0.4
    
    u_weno_plus = np.zeros(Ngrid+1)
    u_weno_minus = np.zeros(Ngrid+1)
    u1 = np.zeros(Ngrid+6)
    u2 = np.zeros(Ngrid+6)

    plt.ion()

    for i in range(Nstep):
        u1 = u.copy()
        u2 = u.copy()
        plt.cla()
        for j in range(Ngrid+1):
            u_weno_minus[j] = weno_reconstruct_minus(u[j:j+5])
            u_weno_plus[j] = weno_reconstruct_plus(u[j+1:j+6])
        dt =  CFL*dx/max(abs(u))
        u1[3:Ngrid+3] = u[3:Ngrid+3]+dt*L(u)
        u2[3:Ngrid+3] = 0.75*u[3:Ngrid+3] + 0.25*u1[3:Ngrid+3] + dt* L(u1)
        u[3:Ngrid+3] = 1/3*u[3:Ngrid+3] + 2/3*u2[3:Ngrid+3] + 2/3*dt*L(u2)
        u[0],u[1],u[2] = u[3]*np.array([1,1,1])
        u[Ngrid+3],u[Ngrid+4],u[Ngrid+5] = u[Ngrid+2]*np.array([1,1,1])
        print(u)

        plt.plot(u)

        plt.pause(1)

    
    plt.ioff()
    plt.show()





    
main()