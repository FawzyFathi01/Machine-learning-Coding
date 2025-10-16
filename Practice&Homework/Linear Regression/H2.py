import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import H1CostFunction as cost


def gradient_descent_linear_regression(X, t, step_size = 0.01, precision = 0.0001, max_iter = 1000):
    # Use step_size = 0.1 and max_iter = 3 (✔️)
    w = np.array([1.0,1.0,1.0,1.0])
    for i in range(max_iter):
        grad = cost.f_dervative(X,t,w)
        w = w - grad * step_size
    return w 






if __name__ == '__main__':
    #read data and Normalize (✔️)
    df = pd.read_csv('dataset_200x4_regression.csv')
    data = np.array(df)
    scalar = MinMaxScaler()
    data_scalar = scalar.fit_transform(data)
    


    # split x | t (✔️)
    x = data_scalar[:,:3]
    t = data_scalar[:,-1]
    
    # add ones to x For W0 (✔️)
    ones = np.ones((x.shape[0],1))
    x = np.hstack([ones,x])
    
    # Optimal weights and cost (✔️) 
    test_max_iter = 100
    for i in range(test_max_iter):    
        optimal_weights = gradient_descent_linear_regression(X=x,t=t,step_size=0.1,max_iter=i)
        print(f"optimal_weights : {optimal_weights}\nCost : {cost.f(x,t,optimal_weights)}")
        print("============================================")




