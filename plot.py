import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
import jax
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import math
from matplotlib.collections import LineCollection
from matplotlib import cm
import matplotlib.animation as animation

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'cm'


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


def ill_singlelayer():
    U_color = np.array([92, 158, 173]) / 255
    V_color = np.array([210, 204, 161]) / 255
    I_color = np.array([239, 111, 108]) / 255
    line_color = np.array([237, 177, 131]) / 255

    def circle(x, y, r, line, color, linewidth, text, size, rx, ry):
        xx = np.linspace(x - r, x + r, 500)
        xxx = np.linspace(x - r, x + r, 500)
        plt.plot(xxx, y + np.sqrt(r ** 2 - (x - xx) ** 2), line, color=color, linewidth=linewidth)
        plt.plot(xxx, y - np.sqrt(r ** 2 - (x - xx) ** 2), line, color=color, linewidth=linewidth)
        plt.text(x - rx, y - ry, text, fontsize=size)

    plt.figure(figsize=(6, 6))
    '''
    circle(1,5,0.5,':','k',2.5,r'$s_1$',30,0.5*0.5,0.5*0.4)
    circle(2.5, 5, 0.5, '-', 'k', 1.5, r'$s_1^{\rm o}$', 30,0.5*0.6,0.5*0.4)
    circle(4, 5, 0.5, '-', I_color, 1.5, r'$I_1^{ext}$', 20,0.5*0.7,0.5*0.4)
    circle(5.5, 5, 0.5, '-', U_color, 1.5, r'$U_1$', 25,0.5*0.6,0.5*0.4)
    circle(7, 5, 0.5, '-', line_color, 1.5, r'$s_1$', 30,0.5*0.5,0.5*0.4)

    circle(1, 3.5, 0.5, ':', 'k', 1.5, r'$s_2$', 30, 0.5*0.5,0.5*0.4)
    circle(2.5, 3.5, 0.5, '-', 'k', 1.5, r'$s_2^{\rm o}$', 30, 0.5 * 0.6, 0.5 * 0.4)
    circle(4, 3.5, 0.5, '-', I_color, 1.5, r'$I_2^{ext}$', 20, 0.5 * 0.7, 0.5 * 0.4)
    circle(5.5, 3.5, 0.5, '-', U_color, 1.5, r'$U_2$', 25, 0.5 * 0.6, 0.5 * 0.4)
    circle(7, 3.5, 0.5, '-', line_color, 1.5, r'$s_2$', 30, 0.5*0.5,0.5*0.4)


    plt.text(0, 4.2, r'$L_{12}$', fontsize=15)
    plt.text(1.4, 4.2, r'$L_{21}$', fontsize=15)
    plt.text(4.5, 4.2, r'$G_{12}$', fontsize=15)
    plt.text(5.9, 4.2, r'$G_{21}$', fontsize=15)
    '''
    circle(1, 3.5, 0.5, ':', 'k', 2.5, r'$s$', 30, 0.5 * 0.4, 0.5 * 0.35)
    circle(2.5, 3.5, 0.5, '-', 'k', 2.5, r'$O$', 30, 0.5 * 0.6, 0.5 * 0.4)
    circle(4, 3.5, 0.5, '-', I_color, 2.5, r'$I^{ext}$', 20, 0.5 * 0.7, 0.5 * 0.4)
    circle(5.5, 3.5, 0.5, '-', U_color, 2.5, r'$U$', 25, 0.5 * 0.5, 0.5 * 0.45)
    circle(7, 3.5, 0.5, '-', line_color, 2.5, r'$s$', 30, 0.5 * 0.4, 0.5 * 0.35)
    S_1 = 10
    plt.text(0.2, 2.6, 'latent feature', fontsize=S_1)
    plt.text(0.7, 2.3, r'$p(s)$', fontsize=S_1)
    plt.text(1.9, 2.6, 'observation', fontsize=S_1)
    plt.text(2.1, 2.3, r'$p(s^{\rm o}|s)$', fontsize=S_1)
    plt.text(3.4, 2.6, 'external input', fontsize=S_1)
    plt.text(5.15, 2.6, 'CANN', fontsize=S_1)
    plt.text(6.2, 2.6, 'bump position', fontsize=S_1)
    plt.text(6.2, 2.3, r'$\tilde{p}(s) = p(s|s^{\rm o})$', fontsize=S_1)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.tight_layout()
    #plt.savefig('./figure/fig1_a.eps')
    plt.savefig('./figure/fig1_a.pdf')
    plt.show()




if __name__ == '__main__':
    ill_singlelayer()
    # eigen(neuron_size = np.array([100] + [100] * 5))
    # eigen(neuron_size=np.array([100] + [50] * 10))
    # eigen(neuron_size=np.array([100] + [10] * 50))
    # eigen(neuron_size = np.array([100] + [5] * 100))
    # eigen(neuron_size=np.array([100] + [2,250,2,250,2]))
    # eigen(neuron_size=np.array([100] + [2, 250, 250]))
    # eigen(neuron_size=np.array([100] + [2, 500, 2]))
    # #eigen(neuron_size=np.array([100] + [2, 250, 2, 125, 125]))
    # #eigen(neuron_size=np.array([1, 2, 1]))
    # #E()
    # a = np.array([2,2,2,2,2,2])
    # b = np.array([1,2,3])
    # print(a[b])