import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as patxi
import random

RED=[255,0,0]
BLUE=[0,255,0]
GREEN=[0,0,255]
WHITE=[255,255,255]


######################################
K=int(input("Enter Penalty Factor: "))
N=50
mg =9.8
tol= N*0.01*mg

###########################################3



'''
Diameter of container will be L
center of the container will be in (10,10)
'''
D=L=20  #Diameter of Container
x_center=10  #X_poistion of center of container
y_center=10 #Y_position of center of container

R_=D/2   # Radius of Container
R=1   #Radius of discs

'''
Class to generate initial random coordinates
of discs inside the circular container
'''
class Solution:

    def __init__(self, radius, x_center, y_center):
        """
        :type radius: float
        :type x_center: float
        :type y_center: float
        """
        self.r = radius
        self.x = x_center
        self.y = y_center

    def randPoint(self):
        """
        :rtype: List[float]
        """
        nr = np.sqrt(random.random()) * self.r
        alpha = random.random() * 2 * 3.141592653
        newx = self.x + nr * np.cos(alpha)
        newy = self.y + nr * np.sin(alpha)
        return [newx, newy]
        


def V(r):
    if r<2*R:
        return 0.5*mg*((r-2*R))**2
    else:
        return 0

def dV(r):
    if r<2*R:
        return mg*((r-2*R))
    else:
        return 0
    
'''
Check if a disc is in contact with the container
'''
def V_wall_modified(r):  # r=np.sqrt(x**2+y**2)
    if r<R:
        return 0.5*mg*((r-R)**2)   # Disc touches "left side" of container
    elif r>R_-R:
        return 0.5*mg*((r+R-R_)**2)    # Disc touches "right side" of container
    else:
        return 0

def dV_wall_modified(r):
    if r<R:
        return mg*((r-R))
    elif r>R_-R:
        return mg*((r+R-R_))
    else:
        return 0 
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def energy(x,penal):
    import numpy as np
    xc=np.array([x_center,y_center])  #Coordinates of container
    n=len(x)
    energy=0.
    for k in range(0,n//2):
        xk=np.array([x[2*k],x[2*k+1]])
        for i in range(k+1,n//2):
            xi=np.array([x[2*i],x[2*i+1]])
            xki=xi-xk
            rki=np.linalg.norm(xki)
            energy+=penal*V(rki)

        energy+=mg*xk[1]
        #pol=np.array([cart2pol(xk[0], xk[1])])
        '''
For each ball we calculate its distance from the center of the container
'''
        xkc=xc-xk      
        rkc=np.linalg.norm(xkc)
        energy+=penal*V_wall_modified(rkc)
        #energy+=penal*V_floor(xk[1])
        #energy+=penal*V_wall(xk[0])
        #add floor and wall contributions to energy
        # energy+=...
    return energy



def d_energy(x,penal):   
    import numpy as np
    n=len(x)
    denergy=np.zeros(n)
    xc=np.array([x_center,y_center])
    for k in range(0,n//2):
        xk=np.array([x[2*k],x[2*k+1]]) 
        for i in range(0,n//2):  
            if (k!=i):
                xi=np.array([x[2*i],x[2*i+1]])                           
                xki=xi-xk                  
                rki=np.linalg.norm(xki)    
                ddV=dV(rki)
                if(ddV!=0):
                    ddV=ddV/rki                                            
                    denergy[2*k]+=-penal*ddV*xki[0]
                    denergy[2*k+1]+=-penal*ddV*xki[1]
        
#For each ball we calculate its distance from the center of the container

        xkc=xc-xk
        rkc=np.linalg.norm(xkc)
        ddwall= dV_wall_modified(rkc)
        if(ddwall!=0):   # We only add the contributions to
                         # vertical and horizontal unit vectors of the gradient
                         #in case ddwall!=0. In other words,we only add the contribution
                         #if the disc touches the container
            ddwall=ddwall/rkc                                            
            denergy[2*k]+=-penal*ddwall*xkc[0]  # The negative sign is because of the rule of chains
            denergy[2*k+1]+=-penal*ddwall*xkc[1]
        
        denergy[2*k+1]+=mg   # Add potential energy contribution
        
    return denergy

    


def line_search_golden(x,p,Vold,fun,alphaold,k):
    '''
α can be fixed (inefficient) or can be obtained by an additional algorithm.
The golden-section search is a technique for finding an extremum (minimum or maximum) of a function inside a specified interval.
The method operates by successively narrowing the range of values on the specified interval, which makes it relatively slow, but very robust. 
'''
    #We start with a random point on the function and move in the
    #negative direction of the gradient of the function to reach the local/global minimum.
    N=len(x)/2.
    r=(np.sqrt(5)-1.)/2.   #The golden ratio
    p_norm=np.max(abs(p))
    alpha_2=(1./p_norm)*N/10.
    alpha_2=min(alpha_2,1E12)
    alpha_1=max(alphaold*.01,alpha_2*1E-6)
    
    alpha_min=alpha_1
    Vmin=Vold
    alpha=alpha_1
    its=0
    
    while (alpha<alpha_2): # increasing alpha
        its+=1
        V=fun(x+alpha*p,k)
        if(V<Vmin):
            alpha_min=alpha
        alpha/=r    
#    if abs(alpha_min-alpha_1)/alpha_1<1E-10: # decrasing alpha
#        print('alpha_1 original',alpha_1) 
#        alpha_2=alpha_1
#        alpha_1=alpha_1/1000.
#        alpha=alpha_1
#        r=0.9
#        while (alpha<alpha_2):
#            its+=1
#            V=fun(x+alpha*p,k)
#            if(V<Vmin):
#                alpha_min=alpha
#            alpha/=r   
#            print ('new alphas in LS',alpha_min,V,Vold,Vmin)
    return alpha_min



def nonlinear_conjugate_gradient(dfun,fun,x0,tol,k):
    x=np.copy(x0)
    res=-1.*dfun(x,k) 
    p=np.copy(res)
    V=fun(x,k)
    All_E=np.array([])
    all_res=np.array([])
    iter_n=np.array([])
    res_scalar=np.linalg.norm(res)
    iter=0
    alpha=1
    while(res_scalar>tol):

        iter=iter+1
        p_old=np.copy(p)      
        res_old=np.copy(res)
        V_old=V
        alpha=line_search_golden(x,p,V_old,fun,alpha,k)
        x = x+alpha*p_old    
        res=-1.*dfun(x,k) 
        V=fun(x,k) 
        res_scalar=np.linalg.norm(res)
        all_res=np.append(all_res,res_scalar)
        curr_E= fun(x,k)
        All_E=np.append(All_E,curr_E)
        iter_n=np.append(iter_n,iter)
        print ('NLCG, iter,res, energy',iter,V,res_scalar)
        if(res_scalar<tol ):
            break
# Several options to choose beta:        
#        beta=np.dot(res,res)/np.dot(res_old,res_old)   
        beta=(np.dot(res,res)-np.dot(res,res_old))/np.dot(res_old,res_old) 
        p=res+beta*p_old  
    return x,all_res,All_E,iter_n






all_E=np.array([])
all_residual= np.array([])
total_iter=np.array([])




'''
Draw the Big container enclosing the discs
'''



fig = plt.figure() 
ax = fig.add_subplot(111) 
circles=[]

c1=patxi.Circle((x_center,y_center),radius=R_,color='white')
   
ax.add_patch(c1)

ax.set_facecolor((1.0, 0.47, 0.42))



plt.xlim([0,20]) 
plt.ylim([0,20])

#title= "N= "+ str(N) +"  K= "+str(K)
#plt.title(title)
plt.xlabel("x")
plt.ylabel("y")

                
xo=set()
'''
Create 50 initial positions
'''

while len(xo)<N:
    sol=Solution(R_,x_center,y_center)

    pol=sol.randPoint()

    x=pol[0]
    y= pol[1]
    temp=set((x,y))
    if temp not in xo:
        xo.add((x,y))

x0=np.array([])        

for pair in xo:
    x0=np.append(x0,pair[0])
    x0=np.append(x0,pair[1])

X,all_residual,all_E,total_iter = nonlinear_conjugate_gradient(d_energy,energy,x0,tol,K)
n=len(X)
print(X)

x=np.array([])
y=np.array([])

for k in range(0,n//2):
    x=np.append(x,X[2*k])
    y=np.append(y,X[2*k+1])

circles=[]
for coord in range(len(x)):
    clor=random.choice(["red","yellow"])
    c1=patxi.Circle((x[coord],y[coord]),radius=1,color=clor)
    
    ax.add_patch(c1)
plt.title("N=50"+" K= "+str(K))
plt.show()
