import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy.stats
from scipy import sparse
import sys
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Sparse-SOR algorithm
def SOR(A, b, n, maxits, eps, w, x):
    sA = sparse.csr_matrix(A)
    val = sA.data
    col = sA.indices
    rowStart = sA.indptr
 
    d = 0
    k = 0
    er = []
    error = eps*2
    sr = 0
    t = 0
    eps_m = 1*10**-16
    tol = 0
    
    if eps < eps_m:
        tol = eps_m
    else:
        tol = eps
    
    error = 10*tol
    
    while error > eps and k < maxits:
        x_new = []
        for i in range(n):
            sum = 0
            for j in range(rowStart[i], rowStart[i+1]):
                sum = sum + val[j]*x[col[j]]   
                if col[j] == i:
                    d = val[j]
            x_new.append(x[i]+w*(b[i]-sum)/d)
        x_new_array = np.array(x_new)
        
        error = np.sqrt(np.sum((abs(a - b))**2 for a, b in zip(x_new_array, x))) # calculating norm
        
        x = x_new_array
        k = k+1
            
    return x

#blackscholes function 
#input: parameters
#output: the last vector of solution and the solution matrix
def blackscholes(strike_price, days, sigma, rate, M, N):
    
    X = strike_price  # price of option at time T
    Smax = 3*X        # maximum value of stock price
    T = days          # number of days to maturity
    r = rate          # risk-free interest rate
    k = T/M/365
    h = Smax/N
    A = np.zeros([N-1,N-1])
    b = np.zeros([N-1,1])
    
    left = np.zeros(M+1) + X
    right = np.zeros(M+1)
    
    #to get the matrix A
    for i in range(N-1):
        for j in range(N-1):
            if i == j:
                A[i,j] = 1 + k*r + k*(sigma**2)*((i+1)**2)
            elif i == j-1:
                A[i,j] = ((-(j)*k)/2)*((j)*(sigma**2) + r)
            elif i == j+1:
                A[i,j] = ((-(i+1)*k)/2)*((i+1)*(sigma**2) - r)
    
    #to get the vector b
    m = M-1
    n = 1
    b = np.zeros([N-1,1])
    x0 = np.zeros([N-1,1])
    
    v = x0
    v[0] = (k/2)*(sigma**2 - r)*X

    #calculate the first b
    while n <= N-1:
        b[n-1] = max((X-n*h),0)
        n += 1
    z = b
    b = b + v
    
            
    #solve for new b using SOR until m=0
    while m != -1:
        x = SOR(A, b, N-1, 100, 10**-16, 1.3, x0)
        z = np.c_[z, x]
        b = x + v
        m -= 1
    z = np.r_[[left],z,[right]] 
    return x,z

strike_price = 10.0
days = 30
X = strike_price

header = ["Strike Price", "Time (days)", "Volatility", "Interest Rate", "Value of N"]
print(''.join(['{0:<20}'.format(w) for w in header]))
output = [strike_price, days, 0.3, 0.02, 12]
print(''.join(['{0:<20}'.format(w) for w in output]))

# to see the effect of solutions when we vary the value of M

for M in [10, 50, 100, 500, 1000]:
    N = 12
    sigma = 0.3
    rate = 0.02
    (f,z) = blackscholes(strike_price, days, sigma, rate, M, N)
    
    f_list = f.flatten().tolist()
    f_list.insert(0,10.0)  # to insert strike price value as option value at n=0
    f_list.append(0.0)     # to insert value 0 when stock price = Smax value
    
    # to plot option fair price (y-axis) against stock price (x-axis)
    
    x = np.linspace(0, 3*X, N+1)  # stock price
    y = f_list                    # option fair price
    plt.plot(x,y,label='M = %s' %M);  
    
    print("")

plt.legend()

# codes for plotting the graphs by varying other parameters (N, rate and sigma)

''''
for N in [5,12,48,72,100]: 
    M = 1000     # M must be large enough for large value of N
    sigma = 0.3
    rate = 0.02
    (f,z) = blackscholes(strike_price, days, sigma, rate, M, N)
    
    f_list = f.flatten().tolist()
    f_list.insert(0,10.0) # to insert strike price value as option value at n=0
    f_list.append(0.0)    # to insert value 0 when stock price = Smax value
    
    # to plot option fair price (y-axis) against stock price (x-axis)
    
    x = np.linspace(0, 3*X, N+1) # stock price
    y = f_list                   # option fair price
    plt.plot(x,y,label='N = %s' %N);
    print("")

plt.legend()

for rate in [0.01,0.02,0.03]:
    N = 48
    M = N*4
    sigma = 0.3
    (f,z) = blackscholes(strike_price, days, sigma, rate, M, N)
    
    f_list = f.flatten().tolist()
    f_list.insert(0,10.0)
    f_list.append(0.0)
    
    header = ["Strike Price", "Time (days)", "Volatility", "Interest Rate", "Value of M", "Value of N"]
    print(''.join(['{0:<20}'.format(w) for w in header]))
    output = [strike_price, days, sigma, rate, M, N]
    print(''.join(['{0:<20}'.format(w) for w in output]))
    
    x = np.linspace(0, 3*X, N+1) 
    y = f_list   
    plt.plot(x,y,label='rate = %s' %rate);
    
    print("")

plt.legend()

for sigma in [0.2,0.3,0.4]:
    N = 48
    M = N*4
    rate = 0.02
    (f,z) = blackscholes(strike_price, days, sigma, rate, M, N)
        
    f_list = f.flatten().tolist()
    f_list.insert(0,10.0)
    f_list.append(0.0)
    
    header = ["Strike Price", "Time (days)", "Volatility", "Interest Rate", "Value of M", "Value of N"]
    print(''.join(['{0:<20}'.format(w) for w in header]))
    output = [strike_price, days, sigma, rate, M, N]
    print(''.join(['{0:<20}'.format(w) for w in output]))
    
    x = np.linspace(0, 3*X, N+1) 
    y = f_list
    plt.plot(x,y,label='sigma = %s' %sigma);
    
    print("")
    
plt.legend()

'''''

########## ----------   3D plotting   ---------- ##########

#set the parameters for the input and scaling the x and y axis
T = 100
M = 200
S = 50
Smax = 3*S
N = 20

fig = plt.figure()
ax = fig.gca(projection='3d')

#scaling the x and y axis 
x = np.linspace(0, T, M+1)
y = np.linspace(0, Smax, N+1)
X, Y = np.meshgrid(x, y)

#get the z axis matrix 
(f,Z) = blackscholes(S, T, 0.3, 0.02, M, N)

#plot the 3D surface vary with Days to Expiry and price of option
ax.plot_surface(X, Y, Z , rstride=8, cstride=8, alpha=0.3)


cset = ax.contour(X, Y, Z, zdir='z', cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='x', cmap=cm.coolwarm)
cset = ax.contour(X, Y, Z, zdir='y',cmap=cm.coolwarm)

#set the x, y, z labels
ax.set_xlabel('Days to Expiry')
ax.set_ylabel('Stock Price')
ax.set_zlabel('price of option')

plt.show()

########## ------------------------------------ ##########