import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
      #path to my scripts
def parab_gen(y,a):
	x = y**2/a
	return x

#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100
y = np.linspace(-4,4,len)

#parab parameters
V = np.array(([0,0],[0,1]))
u = np.array(([-3/2,0]))
f = -9
#p = np.array(([1,0]))
#foc = np.abs(p@u)/2

O = np.array(([0,0]))
#Eigenvalues and eigenvectors
D_vec,P = LA.eig(V)
D = np.diag(D_vec)
p = P[:,0]
eta = 2*u@p
foc = -eta/D_vec[1]
#print(p,foc,D_vec[1])
x = parab_gen(y,foc)
cA = np.vstack((u+eta*p*0.5,V))
cb = np.vstack((-f,(0.5*eta*p-u).reshape(-1,1)))
c = LA.lstsq(cA,cb,rcond=None)[0]
c = c.flatten()
print(c,-29/25,22/25,foc)
c1 = np.array(([(u@V@u-2*D_vec[1]*u@u+D_vec[1]**2*f)/(eta*D_vec[1]**2),0]))
xStandardparab = np.vstack((x,y))
#xActualparab = P@(xStandardparab - c1[:,np.newaxis])-u[:,np.newaxis]/D_vec[1]
xActualparab = P@xStandardparab + c[:,np.newaxis]

#Labeling the coordinates
parab_coords = np.vstack((O,c)).T
plt.scatter(parab_coords[0,:], parab_coords[1,:])
vert_labels = ['$O$','$c$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (parab_coords[0,i], parab_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left

#Plotting the actual parabola
plt.plot(xActualparab[0,:],xActualparab[1,:],label='Parabola',color='g')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
