import numpy as np
from numpy import cos, sin,sqrt
import pandas as pd


m_e = 0.5
c_L = 1
c_R = 2

def analytical_energy (w, n):
    return (n + 0.5)*1*w

def harmonic_potential (w, x):
    return 0.25*(w**2)*(x**2)

def initialize (w, N, x_min, x_max):
    h = (x_max - x_min)/(N - 1)
    
def f (w, x, E):
    V_w = harmonic_potential(w, x)
    return 2*m_e*(V_w - E)

def normalize (phi_raw, N, h_size):
    C_raw = (phi_raw[0]**2) + (phi_raw[-1]**2)
    for i in range(1, N-1):
        if (i % 2) == 0:
            C_raw += 2*(phi_raw[i]**2)
        else:
            C_raw += 4*(phi_raw[i]**2)
    C = h_size*C_raw/3
    phi_norm = np.zeros(N, float)
    phi_norm[:] = phi_raw[:]/(sqrt(C))
    return phi_norm

def delta_calculation (w, h, E, x_min, N):
    phi_L = np.zeros(N, float)
    phi_R = np.zeros(N, float)
    M = N//2
    ##initial value:
    phi_L[1] = c_L
    phi_R[-2] = c_R


    ## f_i
    def f_i (i):
        return f(w, x_min + i * h, E)
    
    def numerov(i):
        return (1 - (h**2)*f_i(i)/12) + 1e-9

    for i in range(2, M + 1):
        phi_L[i] = (1/(numerov(i)))*((h**2)*f_i(i - 1)*phi_L[i - 1] + 2*numerov(i - 1)*phi_L[i - 1] - numerov(i - 2)*phi_L[i - 2])
    
    for i in range(N - 3,  M - 1, -1):
        phi_R[i] = (1/(numerov(i)))*((h**2)*f_i(i + 1)*phi_R[i + 1] + 2*numerov(i+1)*phi_R[i + 1] - numerov(i + 2)*phi_R[i + 2])

    ##print(phi_L)
    ##print(phi_R)
    ##scale
    left_mid = phi_L[M]
    right_mid = phi_R[M]
    scale_ratio = left_mid/right_mid
    phi_R_scaled = np.zeros(N, float)
    phi_R_scaled[:] = scale_ratio*phi_R[:]

    ##calculate the difference:
    left_M = phi_L[M]
    left_M_1 = phi_L[M - 1]
    right_M_1 = phi_R_scaled[M + 1]

    delta = (1/h)*(left_M_1 + right_M_1 - 2*left_M) - h*f_i(M)*left_M
    return delta

def matching_method (w, N, x_min, x_max, E_0, E_1) :
    h = (x_max - x_min)/(N - 1)
    E_small = E_0
    E_large = E_1

    threshold = 1e-7

    def g (E):
        return delta_calculation(w, h, E, x_min, N)
    
    while abs(E_large - E_small) > threshold:
        delta_0 = g(E_small)
        delta_1 = g(E_large)
        #mid point
        E_mid = (E_large + E_small)/2

        delta_mid = g(E_mid)

        if delta_mid * delta_0 < 0:
            E_large = E_mid
        else:
            E_small = E_mid
        
    E_result = E_large

    return E_result

def test(w, N, x_min, x_max):
    h = (x_max - x_min)/(N - 1)
    
    def g (E):
        return delta_calculation(w, h, E, x_min, N)
    coin = g(0) > 0
    for E in np.arange(0, 20, 0.1):
        new = g(E) > 0
        if new ^ coin:
            print(E, new)
            coin = new

def test_2 (w, N, x_min, x_max):
    h = (x_max - x_min)/(N - 1)
    def g (E):
        return delta_calculation(w, h, E, x_min, N)
    g(2.5)

def numerical_solution(w, N, x_min, x_max):
    soln_list = np.zeros(10, float)
    ## n = 0
    soln_list[0] = matching_method(w, N, x_min, x_max, 0.5*w - 0.1, 0.5*w + 0.05)
    ## n = 1
    soln_list[1] = matching_method(w, N, x_min, x_max, 1.5*w - 0.1, 1.5*w + 0.05)
    ## n = 2
    soln_list[2] = matching_method(w, N, x_min, x_max, 2.5*w - 0.1, 2.5*w + 0.05)
    ## n = 3
    soln_list[3] = matching_method(w, N, x_min, x_max, 3.5*w - 0.1, 3.5*w + 0.05)
    ## n = 4
    soln_list[4] = matching_method(w, N, x_min, x_max, 4.5*w - 0.1, 4.5*w + 0.05)
    ## n = 5
    soln_list[5] = matching_method(w, N, x_min, x_max, 5.5*w - 0.1, 5.5*w + 0.05)
    ## n = 6
    soln_list[6] = matching_method(w, N, x_min, x_max, 6.5*w - 0.1, 6.5*w + 0.05)
    ## n = 7
    soln_list[7] = matching_method(w, N, x_min, x_max, 7.5*w - 0.1, 7.5*w + 0.05)
    ## n = 8
    soln_list[8] = matching_method(w, N, x_min, x_max, 8.5*w - 0.1, 8.5*w + 0.05)
    ## n = 9
    soln_list[9] = matching_method(w, N, x_min, x_max, 9.5*w - 0.1, 9.5*w + 0.05)
    return soln_list

def example():
    w = 1
    x_min = -8
    x_max = 8
    N = 100
    numerical_res = numerical_solution(w, N, x_min, x_max)
    analytical_res = np.zeros(10, float)
    for i in range(10):
        analytical_res[i] = analytical_energy(w, i)
    data = {'analytical solution': analytical_res, 'numerical solution': numerical_res}
    df = pd.DataFrame(data=data)
    print(df) 

def main():
    exp = int(input("Type 0 for example result (w = 1) or other number for customized result\n"))
    if exp == 0:
        example()
        return
    w = int(input("Please input an angular frequency value\n"))
    while 1:
        x_min = int(input("Please input the lower bound of x value \n"))
        x_max = int(input("Please input the upper bound of x value \n"))
        if (x_min + x_max) == 0:
            break
        else:
            print("Please give a symmetric bound\n")
    
    N = int(input("Please input the mesh size\n"))
    numerical_res = numerical_solution(w, N, x_min, x_max)
    analytical_res = np.zeros(10, float)
    for i in range(10):
        analytical_res[i] = analytical_energy(w, i)
    data = {'analytical solution': analytical_res, 'numerical solution': numerical_res}
    df = pd.DataFrame(data=data)
    print(df)



if __name__ == '__main__':
    main()