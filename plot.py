import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
import jax

def eigen(neuron_size):
    A = np.eye(np.sum(neuron_size))
    x_index = 0
    y_index = neuron_size[0]
    for i in range(neuron_size.shape[0]-1):
        delta_mu = np.random.normal(0,0.05,(neuron_size[i+1],neuron_size[i]))
        A[y_index:(y_index + neuron_size[i+1]),x_index:(x_index+neuron_size[i])] = delta_mu
        y_index +=  neuron_size[i+1]
        x_index += neuron_size[i]
    A = A.T
    H_0 = A.T@A
    l,_ = np.linalg.eig(H_0)
    print('#####',neuron_size)
    print('lambda_min = ', np.sort(l)[0])
    print('lambda_max = ', np.sort(l)[-1])
    print('det(H_0)=', np.prod(l))
    print('Tr(H_0)=', np.sum(l))

    H_1 = H_0[neuron_size[0]:,neuron_size[0]:]
    l,_ = np.linalg.eig(H_1)
    print('lambda_min = ', np.sort(l)[0])
    print('lambda_max = ', np.sort(l)[-1])
    print('det(H_1)=', np.prod(l))
    print('Tr(H_1)=', np.sum(l))
    #plt.plot(np.arange(sigma.shape[0]), np.square(sigma))
    #plt.loglog(np.arange(sigma.shape[0]), np.square(sigma))
    #plt.show()

def E():
    a = np.random.normal(0,0.05,(1,200))
    print(a)
    U, sigma, V = np.linalg.svd(a)
    print(sigma)
    plt.plot(np.arange(sigma.shape[0]),np.sort(sigma))
    plt.show()


def cal():
    b = bm.array([v for v in range(10)])
if __name__ == '__main__':
    eigen(neuron_size = np.array([100] + [100] * 5))
    eigen(neuron_size=np.array([100] + [50] * 10))
    eigen(neuron_size=np.array([100] + [10] * 50))
    eigen(neuron_size = np.array([100] + [5] * 100))
    eigen(neuron_size=np.array([100] + [2,250,2,250,2]))
    eigen(neuron_size=np.array([100] + [2, 250, 250]))
    eigen(neuron_size=np.array([100] + [2, 500, 2]))
    #eigen(neuron_size=np.array([100] + [2, 250, 2, 125, 125]))
    #eigen(neuron_size=np.array([1, 2, 1]))
    #E()
    a = np.array([2,2,2,2,2,2])
    b = np.array([1,2,3])
    print(a[b])