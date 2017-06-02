
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import scipy.stats
from scipy import sparse
import sys

#Read data from an input file containing n, A and b
#input: file name
#output: the size of matrix A: n, matrix A, matrix b the size of matrix A
def readinputfile(inputfile):
    n = 0
    a = []
    b = []
    with open(inputfile) as fp:
        for i, line in enumerate(fp):
            if i == 0:
                n = int(line)
            elif i >= 1 and i <= n:
                a.append( [ float (x) for x in line.split() ] )
                A = np.array(a)
            elif i == n+1:
                b.append( [ float (x) for x in line.split() ] )
                B = np.array(b)
                b = B.T
    return n,A,b

#Check matrix diagonal values for zeros 
#input: matrix A
#output: True or False
def check_zeros(A):
    D = np.diagonal(A)
    
    if np.count_nonzero(D)==len(D):
        return True # no zeroes on diagonal
    else:
        return False# zeroes on diagonal

    
#convergence check function
#linear equation only converge if the matrix C has spectral radius r(C) < 1
#This function return True when the input matrix A satisfy this condition 
#input: matrix A
#output: True or False
def convergence_check(A):
    
    dnd = 0                           #number of diagonally not dominant row
    
    D = np.diag(np.diag(A))
    U = np.triu(A, k=1)
    L = np.tril(A, k=-1)
    DL = D+L
    DL_inv = np.linalg.inv(DL)
    C = np.dot(DL_inv,U)

    (values,vector) = np.linalg.eig(C)
    v = np.absolute(values)
    v_max = np.amax(v)
    
    # below lines are meant to check matrix A for row diagonally dominant condition
    # dg takes 2 x absolute value of each diagonal entry of A
    # apos takes absolute value of each entry in matrix A
    # the 'for' loop evaluates whether the sum of each row in matrix dg is higher than each row in matrix apos
    # if above condition is not satisfied, diagonally not-dominant 'dnd' will be set not equal to 0
    # dnd != 0 will cause the convergence_check() function to return False
    
    dg = 2*abs(np.diag(np.diag(A)))
    apos = abs(np.array(A)) 
    for i in range(n):
        if dg[i].sum() > apos[i].sum():
            pass
        else:
            dnd += 1
        
    
    if v_max < 1 and dnd == 0 :
        return True                          #solution will converge
    else:
        return False                         #solution will not converge

 
        
# Sparse-SOR algorithm
def SOR(A, b, n, maxits, eps, w, x):
    
    sA = sparse.csr_matrix(A)
    val = sA.data
    col = sA.indices
    rowStart = sA.indptr
    d = 0
    k = 0
    er = []
    sr = 0
    t = 0
    eps_m = 1*10**-16
    tol = 0
    error_list = []
    k_list = []
    
    # below 4-line code is to evaluate given (user-specified) epsilon (eps) against machine epsilon (eps_m)
    # if the specified-epsilon is of higher precision (smaller value) than machine epsilon,
    # then machine eps will be taken as the limiting tolerance
    if eps < eps_m:
        tol = eps_m
    else:
        tol = eps
    
    # below initial error is defined to 10*tol (certainly greater than tol) so that it will run into the while loop
    error = 10*tol
    
    while error > tol and k < maxits:
        x_new = []
        for i in range(n):
            sum = 0
            for j in range(rowStart[i], rowStart[i+1]):
                sum = sum + val[j]*x[col[j]]   
                if col[j] == i:
                    d = val[j]
            x_new.append(x[i]+w*(b[i]-sum)/d)
        x_new_array = np.array(x_new)
        
        #compare norms with appropriate tolerances        
        error = np.sqrt(np.sum((abs(a - b))**2 for a, b in zip(x_new_array, x))) # calculating norm

        ######### --------    divergence check part   -------- #########
        #check divergence: error increasing
        er.append(error)
        if k > 10:              #the error is allowed to increase for the first few iterations 
            if er[k] > er[k-1]: 
                sr = 4          #stop reason: divergence
                break       
        ######### ------------------------------------------- #########
                            
        
        x = x_new_array
        k_list.append(k)
        k = k+1
    
    if k == maxits:
        sr = 1           #stop reason: due to maxits,
    elif sr != 4:
        sr = 2           #stop reason: due to x sequence convergence
   
    return x, k, sr, eps_m, k_list, error_list  


filename_in = input('Enter an input file name: ') or "nas_Sor.in.txt"
n,A,b = readinputfile(filename_in)
# below default epsilon value takes into consideration the standard machine epsilon (eps_m) which was defined earlier
# lower value (higher precision) than the machine epsilon will not be useful as it will be limited by the machine epsilon in such case
eps = 1*10**-16

#solve Ax = b and set stopping reason
if check_zeros(A) == True:
    if convergence_check(A) == True:
        #solve Ax = b using the Sparse-SOR algorithm above
        x, k, sr, eps_m, k_list, error_list = SOR(A, b, n, 100, eps, 0.8, np.zeros((n,1)))
        #plt.scatter(k_list, error_list)    # for plotting the error graph
        
    else:             # did not pass the convergence check 
        sr = 4        # stop reason: divergence
        k = 0
        eps_m = 1*10**-16
else:                  # did not pass zero on diagonal check
    sr = 3             # stop reason: Zero on diagonal
    k = 0              # set the number of iterations to 0
    eps_m = 1*10**-16  # for printing the machine epsilon for output purposes

# write to the output file 
filename_out = input('Enter an output file name: ') or "nas_Sor.out.txt"

sys.stdout = open(filename_out, "w")

#exception handling
if sr == 1:
    stopreason = "Max Iterations reached"
elif sr == 2:
    stopreason = "x Sequence convergence"
elif sr == 3:
    stopreason = "Zero on diagonal"
elif sr == 4:
    stopreason = "x Sequence divergence"
else:
    stopreason = "Cannot proceed"

#format for output file
header = ["Stopping reason", "Max num of iterations", "No. of iterations", "Machine epsilon", "X seq tolerance"]

line2 = [stopreason, 100, k, eps_m, eps]

print(''.join(['{0:<25}'.format(w) for w in header]))
print(''.join(['{0:<25}'.format(w) for w in line2]))


#write vector x to file only if the following condition is met
if sr == 1 or sr == 2:
    print("x = ", x.T)
else:
    pass

sys.stdout.close()

