import numpy as np
import pandas as pd

def f(x,t,w):
    N = len(t)
    y_pred = np.dot(x,w)
    err = y_pred - t
    return (np.sum(err**2)/(2*N))
    # return(np.dot(err.T,err)/2*N) this is Fast

def f_dervative(x,t,w):
    N = len(t)
    y_pred = np.dot(x,w)
    err = y_pred - t
    res = np.dot(x.T,err)
    return res/N
    
if __name__ == '__main__':
    x = np.array([0, 0.2, 0.4, 0.8, 1.0])
    t = x + 5
    x = x.reshape((-1,1))
    x = np.hstack([np.ones((x.shape[0],1)),x])
    w = np.array([1.0,1.0])


    print(f(x,t,w)) # 8 
    print(f_dervative(x,t,w)) # -4 , -1.92


