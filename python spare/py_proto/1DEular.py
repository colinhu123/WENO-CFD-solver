import numpy as np
import matplotlib.pyplot as plt
def visualize_columns(data):
    # Ensure the data has the correct shape
    if data.shape[1] != 6:
        raise ValueError("Input data must have 6 columns.")
    
    # Create a figure with three subplots arranged vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # Plot the 3rd column ([:,3]) in the first subplot
    ax1.plot(data[:, 3], color='r', label='Column 3')
    ax1.set_ylabel('Density')
    #ax1.set_title('Visualization of Column 3')
    ax1.tick_params(axis='y')

    # Plot the 4th column ([:,4]) in the second subplot
    ax2.plot(data[:, 4], color='g', label='Column 4')
    ax2.set_ylabel('velocity')
    #ax2.set_title('Visualization of Column 4')
    ax2.tick_params(axis='y')

    # Plot the 5th column ([:,5]) in the third subplot
    ax3.plot(data[:, 5], color='b', label='Column 5')
    ax3.set_ylabel('pressure')
    #ax3.set_title('Visualization of Column 5')
    ax3.tick_params(axis='y')
    ax3.set_xlabel('Index')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

eps = 1e-6
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

def weno_reconstruct_minus(stencil):
    '''
    Reconstruct the value at i + 1/2 by applying the stencil from I_{i-2} to I_{i+2} 
    Also labelled as Left
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
    Also labelled as Right
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

Ngrid = 1000
dx = 1/Ngrid
N_STEP = 400
u = np.zeros((Ngrid+6,6,N_STEP+1))
CFL = 0.3
'''
0: density
1: flux of momentum
2: flux of energy
3: density
4: velocity
5: pressure
'''



gamma = 1.4
half_index = int((Ngrid+6)/2)
u[:half_index,3,0] = 1.0
u[half_index:,3,0] = 0.125
u[:half_index,5,0] = 1.0
u[half_index:,5,0] = 0.1
u[:,4,0] = 0.0

u[:,0,0] = u[:,3,0]
u[:,1,0] = u[:,3,0]*u[:,4,0]
u[:,2,0] = 0.5*u[:,3,0]*u[:,4,0]**2+u[:,5,0]/(gamma-1)

#visualize_columns(u[:,:,0])



def q_weno(u):
    q = u[:,0:3].copy()
    qR_tmp = np.zeros((Ngrid+1,3))
    qL_tmp = np.zeros((Ngrid+1,3))
    FR_tmp = np.zeros((Ngrid+1,3))
    FL_tmp = np.zeros((Ngrid+1,3))
    for j in range(Ngrid+1):
        for i in range(3):
            qR_tmp[j,i] = weno_reconstruct_plus(q[j+1:j+6,i])
            qL_tmp[j,i] = weno_reconstruct_minus(q[j:j+5,i])
    
    FR_tmp[:,0] = qR_tmp[:,1]
    rhoR = qR_tmp[:,0]
    uR = qR_tmp[:,1]/qR_tmp[:,0]
    eR = qR_tmp[:,2]/qR_tmp[:,0]
    pR = rhoR*(gamma-1)*(eR-0.5*uR**2)
    FR_tmp[:,0] = rhoR*uR
    FR_tmp[:,1] = rhoR*uR**2+pR
    FR_tmp[:,2] = rhoR*uR*(eR+pR/rhoR)

    FL_tmp[:,0] = qL_tmp[:,1]
    rhoL = qL_tmp[:,0]
    uL = qL_tmp[:,1]/qL_tmp[:,0]
    eL = qL_tmp[:,2]/qL_tmp[:,0]
    pL = rhoL*(gamma-1)*(eL-0.5*uL**2)
    FL_tmp[:,0] = rhoL*uL
    FL_tmp[:,1] = rhoL*uL**2+pL
    FL_tmp[:,2] = rhoL*uL*(eL+pL/rhoL)
    return qR_tmp, qL_tmp, FR_tmp, FL_tmp

def Roe(qR,qL,FR,FL,i):
    '''
    Docstring for Roe
    
    :param FR: Flux of right state WENO, length Ngrid+1,3
    :param FL: Flux of left state WENO, length Ngrid+1,3
    '''
    #hR = FR[:,2]/FR[:,0]
    rhoR = qR[:,0]
    uR = qR[:,1]/qR[:,0]
    eR = qR[:,2]/qR[:,0]
    pR = rhoR*(gamma-1)*(eR-0.5*uR**2)
    hR = (qR[:,2] + pR) / qR[:,0]
    #hR = np.divide(FR[:, 2], FR[:, 0], out=np.zeros_like(FR[:, 2]), where=FR[:, 0] != 0)
    rhoR = qR[:,0]
    uR = qR[:,1]/rhoR
    #hL = FL[:,2]/FL[:,0]
    #hL = np.divide(FL[:, 2], FL[:, 0], out=np.zeros_like(FL[:, 2]), where=FL[:, 0] != 0)
    rhoL = qL[:,0]
    uL = qL[:,1]/rhoL
    eL = qL[:,2]/qL[:,0]
    pL = rhoL*(gamma-1)*(eL-0.5*uL**2)
    hL = (qL[:,2] + pL) / qL[:,0]

    u_hat = (uR*np.sqrt(rhoR)+uL*np.sqrt(rhoL))/(np.sqrt(rhoR)+np.sqrt(rhoL))
    h_hat = (hR*np.sqrt(rhoR)+hL*np.sqrt(rhoL))/(np.sqrt(rhoR)+np.sqrt(rhoL))
    a_hat = np.sqrt((gamma-1)*(h_hat-0.5*u_hat**2))
    print(a_hat)
    print(h_hat)
    print(u_hat)
    #file_name_a = "a_hat"+str(i)+".npy"
    phi_hat = np.sqrt(0.5*(gamma-1)*u_hat**2)
    beta_hat = 1/(2*a_hat**2)
    F = np.zeros((Ngrid+1,3))
    for i in range(Ngrid+1):
        
        L = np.array([[1-phi_hat[i]**2/a_hat[i]**2,(gamma-1)*u_hat[i]/a_hat[i]**2,-(gamma-1)/a_hat[i]**2],
                      [phi_hat[i]**2-u_hat[i]*a_hat[i],a_hat[i]-(gamma-1)*u_hat[i],gamma-1],
                      [phi_hat[i]**2+u_hat[i]*a_hat[i],-a_hat[i]-(gamma-1)*u_hat[i],gamma-1]])
        R = np.array([[1,beta_hat[i],beta_hat[i]],
                      [u_hat[i],beta_hat[i]*(u_hat[i]+a_hat[i]),beta_hat[i]*(u_hat[i]-a_hat[i])],
                      [phi_hat[i]**2/(gamma-1),beta_hat[i]*(h_hat[i]+u_hat[i]*a_hat[i]),beta_hat[i]*(h_hat[i]-u_hat[i]*a_hat[i])]])
        A = np.array([[abs(u_hat[i]-a_hat[i]),0,0],
                      [0,abs(u_hat[i]),0],
                      [0,0,abs(u_hat[i]+a_hat[i])]])
        #F[i,:] = 0.5*(FR[i,:]+FL[i,:])-0.5*(u_hat[i]*(u_hat[i]-a_hat[i])*(u_hat[i]+a_hat[i]))*R@L@(qR[i,:]-qL[i,:])
        F[i,:] = 0.5*(FR[i,:]+FL[i,:])-0.5*R@A@L@(qR[i,:]-qL[i,:])

    return F    

def L(u1,i):
    qR,qL,FR,FL = q_weno(u1)
    F = Roe(qR,qL,FR,FL,i)
    return -1/dx*(F[1:,:]-F[:-1,:])

T = 0
t_list = np.array([T])
for i in range(N_STEP):
    qR,qL,FR,FL = q_weno(u[:,0:3,i])
    u[0:3,:,i+1]=u[0:3,:,i]
    u[Ngrid+3:Ngrid+6,:,i+1]=u[Ngrid+3:Ngrid+6,:,i]#BC

    u1 = u[:,0:3,i].copy()
    u2 = u[:,0:3,i].copy()
    c_max = max(np.sqrt(gamma*u[:,5,i]/u[:,3,i]))
    if c_max > 1:
        dt = CFL*dx/c_max
    else:
        dt = CFL*dx/10

    u1[3:Ngrid+3,:] = u[3:Ngrid+3,0:3,i] + dt*L(u[:,0:3,i],i)
    u2[3:Ngrid+3,:] = 0.75*u[3:Ngrid+3,0:3,i] + 0.25*u1[3:Ngrid+3,0:3] + dt* L(u1,i)
    u[3:Ngrid+3,0:3,i+1] = 1/3*u[3:Ngrid+3,0:3,i] + 2/3*u2[3:Ngrid+3,:] + 2/3*dt*L(u2,i)

    u[3:Ngrid+3,3,i+1] = u[3:Ngrid+3,0,i+1]
    u[3:Ngrid+3,4,i+1] = u[3:Ngrid+3,1,i+1]/u[3:Ngrid+3,0,i+1]
    u[3:Ngrid+3,5,i+1] = u[3:Ngrid+3,0,i+1]*(gamma-1)*(u[3:Ngrid+3,2,i+1]/u[3:Ngrid+3,0,i+1]
                                                       -0.5*u[3:Ngrid+3,4,i+1]**2)
    T += dt
    t_list = np.append(t_list,T)
    print(T)

visualize_columns(u[:,:,150])
np.save("SodTestTube.npy",u)
np.save("timeline.npy",t_list)