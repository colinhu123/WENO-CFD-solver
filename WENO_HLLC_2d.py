import numpy as np
import matplotlib.pyplot as plt
from plot_utils import interactive_plot_keyboard

SQRT3_12 = np.sqrt(3)/12

def flux_x(q,gamma):
    '''
    Construct flux from conservative variables
    
    Parameters
    ----------
    q: ndarray of shape (N+1,N,4)
        face averaged conservative variables on walls normal to x_axis

    Returns
    -------
    F: ndarray of shape (N+1,N,4)
        flux average over the cell wall 
    '''
    _eps = 1e-12
    F = np.zeros_like(q)
    rho_safe = np.maximum(_eps,q[:,:,0])
    u = q[:,:,1]/rho_safe
    v = q[:,:,2]/rho_safe
    p = (gamma-1)*(q[:,:,3]-0.5*rho_safe*(u**2+v**2))
    F[:,:,0] = q[:,:,1]
    F[:,:,1] = q[:,:,1]*u+p
    F[:,:,2] = q[:,:,2]*u
    F[:,:,3] = u*(q[:,:,3]+p)

    return F

def flux_y(q,gamma):
    '''
    Construct flux in y-direction from conservative variables
    
    Parameters
    ----------
    q: ndarray of shape (N,N+1,4)
        face average value on walls normal to y-direction
    
    Returns
    -------
    F: ndarray of shape (N,N+1,4)
        flux value over given cells
    '''

    _eps = 1e-12
    F = np.zeros_like(q)
    rho_safe = np.maximum(_eps,q[:,:,0])
    u = q[:,:,1]/rho_safe
    v = q[:,:,2]/rho_safe
    p = (gamma-1)*(q[:,:,3]-0.5*rho_safe*(u**2+v**2))
    F[:,:,0] = q[:,:,2]
    F[:,:,1] = q[:,:,1]*v
    F[:,:,2] = q[:,:,2]*v+p
    F[:,:,3] = v*(q[:,:,3]+p)

    return F

def weno_x(u,gamma):
    '''
    Perform WENO reconstruction on x-axis
    
    Parameters
    ----------
    u: ndarray of shape (N+6,N+6,8)
        cell-average solution values.

    Returns
    -------
    qL: ndarray of shape (N+1,N,4)
        Left-biased reconstruction face average
    qR: ndarray of shape (N+1,N,4)
        Right-biased reconstruction face average
    FL: ndarray of shape (N+1,N,4)
        Left-biased reconstruction flux based on qL
    FR: ndarray of shape (N+1,N,4)
        Right-biased reconstruction flux based on qR
    '''
    # This section of code is for left biased reconstruction
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
    eps = 1e-12
    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    qL0 = (w0*(-u4+5*u3+2*u2)+w1*(-u1+5*u2+2*u3)+w2*(2*u0-7*u1+11*u2))/6
    # The size of qL0 suppose to be (N+1,N+6,4)

    q0 = qL0[:,1:-5,:]
    q1 = qL0[:,2:-4,:]
    q2 = qL0[:,3:-3,:]
    q3 = qL0[:,4:-2,:]
    q4 = qL0[:,5:-1,:]

    beta0 = 13/12*(q2-2*q3+q4)**2+1/4*(3*q2-4*q3+q4)**2
    beta1 = 13/12*(q1-2*q2+q3)**2+1/4*(q1-q3)**2
    beta2 = 13/12*(q0-2*q1+q2)**2+1/4*(q0-4*q1+3*q2)**2

    d0 = 0.1928406937
    d1 = 0.6111111111
    d2 = 0.1960481952

    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    qL1 = w0*(q2+(3*q2-4*q3+q4)*SQRT3_12)+w1*(q2-(-q1+q3)*SQRT3_12)+w2*(q2-(3*q2-4*q1+q0)*SQRT3_12)

    d0 = 0.1960481952
    d1 = 0.6111111111
    d2 = 0.1928406937

    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    qL2 = w0*(q2-(3*q2-4*q3+q4)*SQRT3_12)+w1*(q2-(q1-q3)*SQRT3_12)+w2*(q2-(-3*q2+4*q1-q0)*SQRT3_12)
    
    FL1 = flux_x(qL1,gamma)
    FL2 = flux_x(qL2,gamma)

    FL = 0.5*(FL1+FL2)
    qL = 0.5*(qL1+qL2)

    #This section is for right-biased reconstruction
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
    eps = 1e-12
    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum
    qR0 = (w0*(2*u4-7*u3+11*u2)+w1*(-u3+5*u2+2*u1)+w2*(-u0+5*u1+2*u2))/6

    #begin with high order reconstruciton along the wall.

    q0 = qR0[:,1:-5,:]
    q1 = qR0[:,2:-4,:]
    q2 = qR0[:,3:-3,:]
    q3 = qR0[:,4:-2,:]
    q4 = qR0[:,5:-1,:]

    beta0 = 13/12*(q2-2*q3+q4)**2+1/4*(3*q2-4*q3+q4)**2
    beta1 = 13/12*(q1-2*q2+q3)**2+1/4*(q1-q3)**2
    beta2 = 13/12*(q0-2*q1+q2)**2+1/4*(q0-4*q1+3*q2)**2

    d0 = 0.1928406937
    d1 = 0.6111111111
    d2 = 0.1960481952

    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    qR1 = w0*(q2+(3*q2-4*q3+q4)*SQRT3_12)+w1*(q2-(-q1+q3)*SQRT3_12)+w2*(q2-(3*q2-4*q1+q0)*SQRT3_12)

    d0 = 0.1960481952
    d1 = 0.6111111111
    d2 = 0.1928406937

    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    qR2 = w0*(q2-(3*q2-4*q3+q4)*SQRT3_12)+w1*(q2-(q1-q3)*SQRT3_12)+w2*(q2-(-3*q2+4*q1-q0)*SQRT3_12)
    
    FR1 = flux_x(qR1,gamma)
    FR2 = flux_x(qR2,gamma)

    FR = 0.5*(FR1+FR2)
    qR = 0.5*(qR1+qR2)
    return qL,FL,qR,FR


def weno_y(u,gamma):
    '''
    Reconstruct conservative variable and flux on the wall normal to y-axis
    
    Parameters
    ----------
    u: ndarray of shape (N+6,N+6,4)
        the full scale grid array with ghost cell
    gamma:
        take 1.4 as the feature value of air
    
    Returns
    -------
    qL: ndarray of shape (N+1,N,4)
        Left-biased reconstruction face average
    qR: ndarray of shape (N+1,N,4)
        Right-biased reconstruction face average
    FL: ndarray of shape (N+1,N,4)
        Left-biased reconstruction flux based on qL
    FR: ndarray of shape (N+1,N,4)
        Right-biased reconstruction flux based on qR
    '''
    #This section is for left biased reconstruction
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
    eps = 1e-12
    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    qL0 = (w0*(-u4+5*u3+2*u2)+w1*(-u1+5*u2+2*u3)+w2*(2*u0-7*u1+11*u2))/6

    q0 = qL0[1:-5,:,:]
    q1 = qL0[2:-4,:,:]
    q2 = qL0[3:-3,:,:]
    q3 = qL0[4:-2,:,:]
    q4 = qL0[5:-1,:,:]

    beta0 = 13/12*(q2-2*q3+q4)**2+1/4*(3*q2-4*q3+q4)**2
    beta1 = 13/12*(q1-2*q2+q3)**2+1/4*(q1-q3)**2
    beta2 = 13/12*(q0-2*q1+q2)**2+1/4*(q0-4*q1+3*q2)**2

    d0 = 0.1928406937
    d1 = 0.6111111111
    d2 = 0.1960481952

    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    qL1 = w0*(q2+(3*q2-4*q3+q4)*SQRT3_12)+w1*(q2-(-q1+q3)*SQRT3_12)+w2*(q2-(3*q2-4*q1+q0)*SQRT3_12)

    d0 = 0.1960481952
    d1 = 0.6111111111
    d2 = 0.1928406937

    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    qL2 = w0*(q2-(3*q2-4*q3+q4)*SQRT3_12)+w1*(q2-(q1-q3)*SQRT3_12)+w2*(q2-(-3*q2+4*q1-q0)*SQRT3_12)

    FL1 = flux_y(qL1,gamma)
    FL2 = flux_y(qL2,gamma)

    FL = 0.5*(FL1+FL2)
    qL = 0.5*(qL1+qL2)

    #This section is for right-biased reconstruction

    u0 = u_con[:,1:-4,:]
    u1 = u_con[:,2:-3,:]
    u2 = u_con[:,3:-2,:]
    u3 = u_con[:,4:-1,:]
    u4 = u_con[:,5:,:]

    beta0 = 13/12*(u2-2*u3+u4)**2+1/4*(3*u2-4*u3+u4)**2
    beta1 = 13/12*(u1-2*u2+u3)**2+1/4*(u1-u3)**2
    beta2 = 13/12*(u0-2*u1+u2)**2+1/4*(u0-4*u1+3*u2)**2

    d0 = 0.1
    d1 = 0.6
    d2 = 0.3
    eps = 1e-12
    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum
    qR0 = (w0*(2*u4-7*u3+11*u2)+w1*(-u3+5*u2+2*u1)+w2*(-u0+5*u1+2*u2))/6

    q0 = qR0[1:-5,:,:]
    q1 = qR0[2:-4,:,:]
    q2 = qR0[3:-3,:,:]
    q3 = qR0[4:-2,:,:]
    q4 = qR0[5:-1,:,:]

    beta0 = 13/12*(q2-2*q3+q4)**2+1/4*(3*q2-4*q3+q4)**2
    beta1 = 13/12*(q1-2*q2+q3)**2+1/4*(q1-q3)**2
    beta2 = 13/12*(q0-2*q1+q2)**2+1/4*(q0-4*q1+3*q2)**2

    d0 = 0.1928406937
    d1 = 0.6111111111
    d2 = 0.1960481952

    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    qR1 = w0*(q2+(3*q2-4*q3+q4)*SQRT3_12)+w1*(q2-(-q1+q3)*SQRT3_12)+w2*(q2-(3*q2-4*q1+q0)*SQRT3_12)

    d0 = 0.1960481952
    d1 = 0.6111111111
    d2 = 0.1928406937

    a0 = d0/(eps+beta0)**2
    a1 = d1/(eps+beta1)**2
    a2 = d2/(eps+beta2)**2

    o_sum = a0+a1+a2
    w0 = a0/o_sum
    w1 = a1/o_sum
    w2 = a2/o_sum

    qR2 = w0*(q2-(3*q2-4*q3+q4)*SQRT3_12)+w1*(q2-(q1-q3)*SQRT3_12)+w2*(q2-(-3*q2+4*q1-q0)*SQRT3_12)

    FR1 = flux_y(qR1,gamma)
    FR2 = flux_y(qR2,gamma)

    FR = 0.5*(FR1+FR2)
    qR = 0.5*(qR1+qR2)

    return qL,FL,qR,FR

def con2primi(q,gamma:float):
    '''
    This function is to convert conservative variable to primitive variable,
    including pressure, sound speed, density,u,v in safe form
    
    Parameters
    ----------
    u: ndarray of any shape, (N+1,N,4) or (N,N+1,4)
        conservative variable in the given point
    gamma: f64
        feature thermodynamic constant

    Returns
    -------
    p: ndarray of shape (N+1,N) or (N,N+1)
        pressure at all points with positive value check
    a: ndarray of shape (N+1,N) or (N,N+1)
        local acoustic speed at all points
    rho_safe: ndarray of shape (N+1,N) or (N,N+1)
        positive check density
    u: ndarray of shape (N+1,N) or (N,N+1)
        velocity in x-direction
    v: ndarray of shape (N+1,N) or (N,N+1)
        velocity in y-direction
    '''
    _eps = 1e-12
    rho_safe = np.maximum(q[:,:,0],_eps)
    u = q[:,:,1]/rho_safe
    v = q[:,:,2]/rho_safe
    p = (gamma-1)*(q[:,:,3]-0.5*rho_safe*(u**2+v**2))
    p = np.maximum(p,_eps)
    a = np.sqrt(gamma*p/rho_safe)
    return p,a,rho_safe,u,v


def HLLC_x(qL, qR, F_L, F_R, gamma):
    """
    HLLC flux in x-direction (Toro formulation).
    qL, qR, F_L, F_R shape: (N+1,N,4)
    Returns: F shape (N+1,N,4)
    """
    _eps = 1e-12
    # primitive conversion (ensure con2primi is correct!)
    pL, aL, rhoL, uL, vL = con2primi(qL, gamma)
    pR, aR, rhoR, uR, vR = con2primi(qR, gamma)
    HL = (qL[:,:,3]+pL)/rhoL
    HR = (qR[:,:,3]+pR)/rhoR
    # wave speed estimates (simple Davis/Einfeldt choice)
    #SL = np.minimum(uL - aL, uR - aR)
    #SR = np.maximum(uL + aL, uR + aR)
    u_tilde = (np.sqrt(rhoL)*uL+np.sqrt(rhoR)*uR)/(np.sqrt(rhoL)+np.sqrt(rhoR))
    v_tilde = (np.sqrt(rhoL)*vL+np.sqrt(rhoR)*vR)/(np.sqrt(rhoL)+np.sqrt(rhoR))
    H_tilde = (np.sqrt(rhoL)*HL+np.sqrt(rhoR)*HR)/(np.sqrt(rhoL)+np.sqrt(rhoR))
    a_tilde = np.sqrt((gamma-1)*(H_tilde-0.5*(u_tilde**2+v_tilde**2)))
    SL= u_tilde-a_tilde
    SR = u_tilde+a_tilde
    # prevent degenerate denominator
    denom = rhoL * (SL - uL) - rhoR * (SR - uR)
    denom = np.where(np.abs(denom) < _eps, _eps * np.sign(denom) + _eps, denom)

    # Toro eq for S*
    Sstar = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / denom

    # Compute conserved Qstar for left and right using Toro eq (39)/(71)
    # Left star conserved state
    coeffL = rhoL * (SL - uL) / (SL - Sstar)
    # specific total energy for qL: E/rho
    eL_spec = qL[..., 3] / rhoL
    Estar_spec_L = eL_spec + (Sstar - uL) * (Sstar + pL / (rhoL * (SL - uL)))
    QstarL = np.empty_like(qL)
    QstarL[..., 0] = coeffL
    QstarL[..., 1] = coeffL * Sstar
    QstarL[..., 2] = coeffL * vL
    QstarL[..., 3] = coeffL * Estar_spec_L

    # Right star conserved state
    coeffR = rhoR * (Sstar - uR) / (SR - Sstar)
    eR_spec = qR[..., 3] / rhoR
    Estar_spec_R = eR_spec + (Sstar - uR) * (Sstar + pR / (rhoR * (SR - uR)))
    QstarR = np.empty_like(qR)
    QstarR[..., 0] = coeffR
    QstarR[..., 1] = coeffR * Sstar
    QstarR[..., 2] = coeffR * vR
    QstarR[..., 3] = coeffR * Estar_spec_R

    # Now compute flux picking according to wave speeds (Toro eq (26),(27),(29))
    F = np.empty_like(qL)

    # mask where left-going everything (SL >= 0)
    mask_L = SL >= 0
    if np.any(mask_L):
        F[mask_L] = F_L[mask_L]

    # SL < 0 and Sstar >= 0 -> star-left region
    mask_Ls = (SL < 0) & (Sstar >= 0)
    if np.any(mask_Ls):
        F[mask_Ls] = F_L[mask_Ls] + SL[mask_Ls][..., None] * (QstarL[mask_Ls] - qL[mask_Ls])

    # Sstar < 0 and SR > 0 -> star-right region
    mask_sR = (Sstar < 0) & (SR > 0)
    if np.any(mask_sR):
        F[mask_sR] = F_R[mask_sR] + SR[mask_sR][..., None] * (QstarR[mask_sR] - qR[mask_sR])

    # right-going everything (SR <= 0)
    mask_R = SR <= 0
    if np.any(mask_R):
        F[mask_R] = F_R[mask_R]

    return F

def HLLC_y(qL, qR, F_L, F_R, gamma):
    """
    HLLC flux in y-direction (Toro formulation).
    qL, qR, F_L, F_R shape: (N,N+1,4) — note: left/right are in y-normal sense.
    Returns: F shape (N,N+1,4)
    """
    _eps = 1e-12
    pL, aL, rhoL, uL, vL = con2primi(qL, gamma)
    pR, aR, rhoR, uR, vR = con2primi(qR, gamma)

    # wave speeds in y-direction use v as normal
    #SL = np.minimum(vL - aL, vR - aR)
    #SR = np.maximum(vL + aL, vR + aR)
    HL = (qL[:,:,3]+pL)/rhoL
    HR = (qR[:,:,3]+pR)/rhoR
    u_tilde = (np.sqrt(rhoL)*uL+np.sqrt(rhoR)*uR)/(np.sqrt(rhoL)+np.sqrt(rhoR))
    v_tilde = (np.sqrt(rhoL)*vL+np.sqrt(rhoR)*vR)/(np.sqrt(rhoL)+np.sqrt(rhoR))
    H_tilde = (np.sqrt(rhoL)*HL+np.sqrt(rhoR)*HR)/(np.sqrt(rhoL)+np.sqrt(rhoR))
    a_tilde = np.sqrt((gamma-1)*(H_tilde-0.5*(u_tilde**2+v_tilde**2)))
    SL= v_tilde-a_tilde
    SR = v_tilde+a_tilde
    denom = rhoL * (SL - vL) - rhoR * (SR - vR)
    denom = np.where(np.abs(denom) < _eps, _eps * np.sign(denom) + _eps, denom)

    Sstar = (pR - pL + rhoL * vL * (SL - vL) - rhoR * vR * (SR - vR)) / denom

    # Left star conserved state (note ordering: [rho, rho*u, rho*v, E])
    coeffL = rhoL * (SL - vL) / (SL - Sstar)
    eL_spec = qL[..., 3] / rhoL
    Estar_spec_L = eL_spec + (Sstar - vL) * (Sstar + pL / (rhoL * (SL - vL)))
    QstarL = np.empty_like(qL)
    QstarL[..., 0] = coeffL
    QstarL[..., 1] = coeffL * uL    # tangential velocity here
    QstarL[..., 2] = coeffL * Sstar
    QstarL[..., 3] = coeffL * Estar_spec_L

    # Right star
    coeffR = rhoR * (Sstar - vR) / (SR - Sstar)
    eR_spec = qR[..., 3] / rhoR
    Estar_spec_R = eR_spec + (Sstar - vR) * (Sstar + pR / (rhoR * (SR - vR)))
    QstarR = np.empty_like(qR)
    QstarR[..., 0] = coeffR
    QstarR[..., 1] = coeffR * uR
    QstarR[..., 2] = coeffR * Sstar
    QstarR[..., 3] = coeffR * Estar_spec_R

    F = np.empty_like(qL)

    mask_L = SL >= 0
    if np.any(mask_L):
        F[mask_L] = F_L[mask_L]

    mask_Ls = (SL < 0) & (Sstar >= 0)
    if np.any(mask_Ls):
        F[mask_Ls] = F_L[mask_Ls] + SL[mask_Ls][..., None] * (QstarL[mask_Ls] - qL[mask_Ls])

    mask_sR = (Sstar < 0) & (SR > 0)
    if np.any(mask_sR):
        F[mask_sR] = F_R[mask_sR] + SR[mask_sR][..., None] * (QstarR[mask_sR] - qR[mask_sR])

    mask_R = SR <= 0
    if np.any(mask_R):
        F[mask_R] = F_R[mask_R]

    return F

def HLLE_x(qL, qR, F_L, F_R, gamma):
    """
    HLLE flux in x-direction (face normal = x).
    qL, qR, F_L, F_R shape: (N+1,N,4)
    Returns: F shape (N+1,N,4)
    """
    _eps = 1e-12
    pL, aL, rhoL, uL, vL = con2primi(qL, gamma)
    pR, aR, rhoR, uR, vR = con2primi(qR, gamma)

    # same Roe-like / Einfeldt tilde estimates you used
    HL = (qL[...,3] + pL) / rhoL
    HR = (qR[...,3] + pR) / rhoR
    u_tilde = (np.sqrt(rhoL)*uL + np.sqrt(rhoR)*uR) / (np.sqrt(rhoL) + np.sqrt(rhoR))
    v_tilde = (np.sqrt(rhoL)*vL + np.sqrt(rhoR)*vR) / (np.sqrt(rhoL) + np.sqrt(rhoR))
    H_tilde = (np.sqrt(rhoL)*HL + np.sqrt(rhoR)*HR) / (np.sqrt(rhoL) + np.sqrt(rhoR))
    a_tilde = np.sqrt(np.maximum(0.0, (gamma - 1.0)*(H_tilde - 0.5*(u_tilde**2 + v_tilde**2))))

    SL = u_tilde - a_tilde
    SR = u_tilde + a_tilde

    # ensure denominator not zero when used in intermediate formula
    denom = SR - SL
    denom = np.where(np.abs(denom) < _eps, _eps * np.sign(denom) + _eps, denom)

    # allocate flux
    F = np.empty_like(qL)

    mask_L = SL >= 0
    if np.any(mask_L):
        F[mask_L] = F_L[mask_L]

    mask_R = SR <= 0
    if np.any(mask_R):
        F[mask_R] = F_R[mask_R]

    # HLL middle region
    mask_middle = (~mask_L) & (~mask_R)
    if np.any(mask_middle):
        SLm = SL[mask_middle][..., None]
        SRm = SR[mask_middle][..., None]
        qL_m = qL[mask_middle]
        qR_m = qR[mask_middle]
        FL_m = F_L[mask_middle]
        FR_m = F_R[mask_middle]
        F[mask_middle] = (SRm * FL_m - SLm * FR_m + SRm * SLm * (qR_m - qL_m)) / (SRm - SLm)

    return F

def HLLE_y(qL, qR, F_L, F_R, gamma):
    """
    HLLE flux in y-direction (face normal = y).
    qL, qR, F_L, F_R shape: (N,N+1,4)
    Returns: F shape (N,N+1,4)
    """
    _eps = 1e-12
    pL, aL, rhoL, uL, vL = con2primi(qL, gamma)
    pR, aR, rhoR, uR, vR = con2primi(qR, gamma)

    HL = (qL[...,3] + pL) / rhoL
    HR = (qR[...,3] + pR) / rhoR
    u_tilde = (np.sqrt(rhoL)*uL + np.sqrt(rhoR)*uR) / (np.sqrt(rhoL) + np.sqrt(rhoR))
    v_tilde = (np.sqrt(rhoL)*vL + np.sqrt(rhoR)*vR) / (np.sqrt(rhoL) + np.sqrt(rhoR))
    H_tilde = (np.sqrt(rhoL)*HL + np.sqrt(rhoR)*HR) / (np.sqrt(rhoL) + np.sqrt(rhoR))
    a_tilde = np.sqrt(np.maximum(0.0, (gamma - 1.0)*(H_tilde - 0.5*(u_tilde**2 + v_tilde**2))))

    # For y-flux use v as normal
    SL = v_tilde - a_tilde
    SR = v_tilde + a_tilde

    denom = SR - SL
    denom = np.where(np.abs(denom) < _eps, _eps * np.sign(denom) + _eps, denom)

    F = np.empty_like(qL)

    mask_L = SL >= 0
    if np.any(mask_L):
        F[mask_L] = F_L[mask_L]

    mask_R = SR <= 0
    if np.any(mask_R):
        F[mask_R] = F_R[mask_R]

    mask_middle = (~mask_L) & (~mask_R)
    if np.any(mask_middle):
        SLm = SL[mask_middle][..., None]
        SRm = SR[mask_middle][..., None]
        qL_m = qL[mask_middle]
        qR_m = qR[mask_middle]
        FL_m = F_L[mask_middle]
        FR_m = F_R[mask_middle]
        F[mask_middle] = (SRm * FL_m - SLm * FR_m + SRm * SLm * (qR_m - qL_m)) / (SRm - SLm)

    return F

def shock_sensor(pR,pL,Jp_lo=0.15,Jp_hi=0.4):
    _eps = 1e-12
    Jp = np.abs(pR - pL) / (np.maximum(pR, pL) + _eps)

    # linear smooth blending weight phi in [0,1]
    phi = np.where(Jp <= Jp_lo, 1.0,
                   np.where(Jp >= Jp_hi, 0.0,
                            1.0 - (Jp - Jp_lo) / (Jp_hi - Jp_lo)))
    
    return phi

def shock_sensor_grad_p(pR,pL,Jp_lo=0.15,Jp_hi=0.4):
    _eps = 1e-12
    (Nx,Ny) = np.shape(pR)
    dx = 1/min(Nx,Ny)
    p = 0.5*(pL+pR)
    dpdx = np.zeros_like(pR)
    dpdy = np.zeros_like(pR)

    dpdx[1:-1,:] = (p[2:,:]-p[:-2,:])/(2*dx)
    dpdx[0,:] = (p[1,:]-p[0,:])/(2*dx)
    dpdx[-1,:] = (p[-1,:]-p[-2,:])/(2*dx)
    dpdy[:,1:-1] = (p[:,2:]-p[:,:-2])/(2*dx)
    dpdy[:,0] =  (p[:,1]-p[:,0])/(2*dx)
    dpdy[:,-1] = (p[:,-1]-p[:,-2])/(2*dx)

    grad_p = np.sqrt(dpdx**2+dpdy**2)
    Jp = grad_p/p

    phi = np.where(Jp <= Jp_lo, 1.0,
                    np.where(Jp >= Jp_hi, 0.0,
                            1.0 - (Jp - Jp_lo) / (Jp_hi - Jp_lo)))
    
    return phi

def HLLC_HLLE_x(qL, qR, F_L, F_R, gamma, Jp_lo=0.15, Jp_hi=0.4):
    """
    Face-based blended flux in x-direction.
    Blends HLLC (phi=1) with HLLE (phi=0) using pressure jump:
      Jp = |pR - pL| / max(pR,pL)
    Linear smooth blend between Jp_lo and Jp_hi.
    """
    _eps = 1e-12
    pL, aL, rhoL, uL, vL = con2primi(qL, gamma)
    pR, aR, rhoR, uR, vR = con2primi(qR, gamma)

    phi = shock_sensor(pR,pL,Jp_lo,Jp_hi)

    F_hllc = HLLC_x(qL, qR, F_L, F_R, gamma)
    F_hlle = HLLE_x(qL, qR, F_L, F_R, gamma)

    phi_exp = phi[..., None]  # expand for vector components
    F = phi_exp * F_hllc + (1.0 - phi_exp) * F_hlle
    return F

def HLLC_HLLE_y(qL, qR, F_L, F_R, gamma, Jp_lo=0.15, Jp_hi=0.4):
    """
    Face-based blended flux in y-direction (normal = y).
    """
    _eps = 1e-12
    pL, aL, rhoL, uL, vL = con2primi(qL, gamma)
    pR, aR, rhoR, uR, vR = con2primi(qR, gamma)

    phi = shock_sensor_grad_p(pR,pL,Jp_lo,Jp_hi)

    F_hllc = HLLC_y(qL, qR, F_L, F_R, gamma)
    F_hlle = HLLE_y(qL, qR, F_L, F_R, gamma)

    phi_exp = phi[..., None]
    F = phi_exp * F_hllc + (1.0 - phi_exp) * F_hlle
    return F

def L(q,dx,gamma,force_HLLE):
    qLx,FLx,qRx,FRx = weno_x(q,gamma)
    qLy,FLy,qRy,FRy = weno_y(q,gamma)
    if force_HLLE:
        F_x = HLLE_x(qLx,qRx,FLx,FRx,gamma)
        F_y = HLLE_y(qLy,qRy,FLy,FRy,gamma)
    else:
        F_x = HLLC_HLLE_x(qLx,qRx,FLx,FRx,gamma,Jp_lo=0.4,Jp_hi=1.0)
        F_y = HLLC_HLLE_y(qLy,qRy,FLy,FRy,gamma,Jp_lo=0.4,Jp_hi=1.0)
    df_dx = (F_x[1:,:,:]-F_x[:-1,:,:])/dx
    dg_dy = (F_y[:,1:,:]-F_y[:,:-1,:])/dx

    return -(df_dx+dg_dy)

def apply_bc_corrected(u, Ngrid):

    i_start, i_end = 3, 3 + Ngrid
    j_start, j_end = 3, 3 + Ngrid

    for i in [0, 1, 2]:
        u[i, :, :] = u[3, :, :]

    for i in [Ngrid+3, Ngrid+4, Ngrid+5]:
        u[i, :, :] = u[Ngrid+2, :, :]
    for j in [0, 1, 2]:
        u[:, j, :] = u[:, 3, :]
    for j in [Ngrid+3, Ngrid+4, Ngrid+5]:
        u[:, j, :] = u[:, Ngrid+2, :]
    
    return u

def RK3(u,Ngrid,dt):
    u1 = u.copy()
    u2 = u.copy()
    u1 = apply_bc_corrected(u1,Ngrid)
    u2[3:Ngrid+3,3:Ngrid+3,:4] = 0.75*u[3:Ngrid+3,3:Ngrid+3,:4]+0.25*u1[3:Ngrid+3,3:Ngrid+3,:4]+dt*L(u1,1/Ngrid,gamma=1.4,force_HLLE=False)
    u2 = apply_bc_corrected(u2,Ngrid)
    u[3:Ngrid+3,3:Ngrid+3,:4] = 1/3*u[3:Ngrid+3,3:Ngrid+3,:4] + 2/3*u2[3:Ngrid+3,3:Ngrid+3,:4] + 2/3*dt*L(u2,1/Ngrid,gamma = 1.4,force_HLLE=False)
    u = apply_bc_corrected(u,Ngrid)
    return u

def RK3_correct(u, Ngrid, dt, gamma,force_HLLE):
    # 第一步
    L0 = L(u, 1/Ngrid, gamma,force_HLLE)
    u1 = u.copy()
    u1[3:Ngrid+3, 3:Ngrid+3, :4] = u[3:Ngrid+3, 3:Ngrid+3, :4] + dt * L0
    u1 = apply_bc_corrected(u1, Ngrid)
    
    # 第二步
    L1 = L(u1, 1/Ngrid, gamma,force_HLLE)
    u2 = u.copy()
    u2[3:Ngrid+3, 3:Ngrid+3, :4] = (
        0.75 * u[3:Ngrid+3, 3:Ngrid+3, :4] + 
        0.25 * u1[3:Ngrid+3, 3:Ngrid+3, :4] + 
        0.25 * dt * L1
    )
    u2 = apply_bc_corrected(u2, Ngrid)
    
    # 第三步
    L2 = L(u2, 1/Ngrid, gamma,force_HLLE)
    u_next = u.copy()
    u_next[3:Ngrid+3, 3:Ngrid+3, :4] = (
        1/3 * u[3:Ngrid+3, 3:Ngrid+3, :4] + 
        2/3 * u2[3:Ngrid+3, 3:Ngrid+3, :4] + 
        2/3 * dt * L2
    )
    return apply_bc_corrected(u_next, Ngrid)

def con_var2primi_pfloor(u,gamma):
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


def main(Ngrid,N_STEP,fileStorage = True,force_HLLE = False):
    import datetime
    import os
    u = np.ones((Ngrid+6,Ngrid+6,8))
    full_path = ""
    folder_name = ""

    gamma = 1.4
    dx = 1/Ngrid

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
    
    if fileStorage:
        current_time = datetime.datetime.now()
        folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        full_path = os.path.join("data", folder_name)
        os.makedirs(full_path,exist_ok=True)
        np.save(os.path.join(full_path,"0.npy"),u)
    t_list = np.array([0])
    T = 0
    for i in range(1,N_STEP):
        print(i)
        speed = np.sqrt(u[3:Ngrid+3,3:Ngrid+3,5]**2+u[3:Ngrid+3,3:Ngrid+3,6]**2)
        c_max = np.max(np.sqrt(gamma*u[3:Ngrid+3,3:Ngrid+3,7]/np.maximum(u[3:Ngrid+3,3:Ngrid+3,0],1e-10))+speed)

        dt =  CFL*dx/c_max

        u = RK3_correct(u,Ngrid,dt,gamma,force_HLLE)

        u = con_var2primi_pfloor(u,gamma)
        T += dt
        if fileStorage:
            np.save(os.path.join(full_path,f"{i}.npy"),u)
        
        t_list = np.append(t_list,T)
    if fileStorage:
        np.save(os.path.join(full_path,"time.npy"),t_list)
        interactive_plot_keyboard(full_path,initial_step=0,var='rho')


main(200,400,fileStorage = True,force_HLLE=False)

