import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
import math
import jax
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2
import os
from matplotlib.animation import ArtistAnimation
from functools import partial
from PIL import Image
from main import PCN, sample, cos_anneal, plot_neuron_2

size = 200
l_bound = 0
r_bound = 1
x = np.linspace(l_bound, r_bound, size + 1)[0:-1]
y = np.linspace(l_bound, r_bound, size + 1)[0:-1]
X, Y = np.meshgrid(x, y)

def generate_pdf(bump_num = 8):
    bump_num = 4
    # center = np.random.rand(bump_num,2)*(r_bound-l_bound)/2 + l_bound + (r_bound-l_bound)/4
    # Lambda = np.random.rand(2,2,bump_num)*10
    center = np.array([[0.2,0.2,0.8,0.8],[0.2,0.8,0.2,0.8]]).T
    Lambda = np.eye(2) * 10
    pdf = np.zeros((size,size))
    for i in range(bump_num):
        delta = np.einsum('wd,dnm->wnm',Lambda[:,:],np.stack((X-center[i,0],Y-center[i,1])))
        Z = np.exp(-delta[0,:,:]**2-delta[1,:,:]**2)
        pdf += Z
    pdf = pdf/np.sum(pdf)
    plt.figure()
    plt.imshow(pdf,origin = 'lower')
    return pdf

def get_prob(pdf,i,j):
    return pdf[int(i*size),int(j*size)]

def generate_data(pdf):
    data = []
    base = size*size
    for i in x:
        for j in y:
            num = int(get_prob(pdf,i,j) * base)
            data += [[i,j]] * num
    data = np.array(data)
    np.save('./data/gdata.npy',data)
    # plt.figure()
    # plt.scatter(data[:,0],data[:,1])
    # plt.title('generate')
    # plt.xlim([l_bound,r_bound])
    # plt.ylim([l_bound, r_bound])
    return data

def show_pdf(pdf_name,data_name = None, data = None):
    if data_name != None:
        data = np.load(data_name)
    pdf = np.zeros((size,size))
    for i in range(data.shape[0]):
        if int(data[i,0]*size) < size and int(data[i,1]*size) < size:
            pdf[int(data[i,0]*size),int(data[i,1]*size)] += 1
    pdf /= np.sum(pdf)
    plt.figure()
    plt.imshow(pdf, origin = 'lower')
    plt.title(pdf_name)


def sample_data(pdf,m):
    def get_grad(pdf,x,y):
        grad_x = np.log(get_prob(pdf, x + (1 / size), y)) - np.log(get_prob(pdf, x, y))
        grad_y = np.log(get_prob(pdf, x, y + (1 / size))) - np.log(get_prob(pdf, x, y))
        return np.array([grad_x,grad_y])*size

    def lim(z):
        for i in range(2):
            if z[i] > r_bound - (2 / size):
                z[i] = r_bound - (2 / size)
            if z[i] < l_bound:
                z[i] = l_bound
        return z

    T = 100000
    z = np.zeros((T,2))
    v = np.zeros((T,2))
    dt = 0.05
    tau = 10000
    noise = [[0,0]]
    for i in range(T - 1):
        tmp_noise1 = np.random.normal(0, 2,size = (2,))
        tmp_noise2 = np.random.normal(0, 2, size=(2,))
        dz = (get_grad(pdf,z[i,0],z[i,1]) - v[i,:]) * (dt / tau) + tmp_noise1 * np.sqrt(dt / tau)
        dv = -m * v[i] * dt + 2 * m * tmp_noise1 * np.sqrt(dt * tau)
        z[i + 1,:] = lim(z[i,:] + dz)
        #z[i + 1, :] = z[i, :] + dz
        v[i + 1,:] = v[i,:] + dv
        noise.append(tmp_noise1)
    np.save('./data/sample_data.npy',z)
    # plt.figure()
    # plt.scatter(z[:,0],z[:,1])
    # plt.xlim([l_bound,r_bound])
    # plt.ylim([l_bound, r_bound])
    return z,v,np.array(noise)


if __name__ == '__main__':
    pdf = generate_pdf()
    data = generate_data(pdf)

    data = np.array([[0.2,0.2]]*100 + [[0.2,0.8]]*100 + [[0.8,0.2]]*100 + [[0.8,0.8]]*100)
    data_length = data.shape[0]
    print(data.shape)
    # #generate_data(pdf)
    # #show_pdf('./data/gdata.npy')
    # z = sample_data(pdf,m = 0)
    # show_pdf('./data/sample_data.npy','m = 0')
    # z,v,noise = sample_data(pdf,m = 20)
    # print(noise.shape)
    # plt.figure()
    # length = 1000
    # plt.plot(np.linspace(0,1,length),-v[0:length,0] * 0.0001 + noise[0:length,0] * np.sqrt(0.0001))
    # plt.plot(np.linspace(0,1,length), noise[0:length,0] * np.sqrt(0.0001))
    # show_pdf('./data/sample_data.npy','m = 10')
    # plt.show()

    Epoch = 30
    using_epoch = 29
    train = True


    learning_rate = cos_anneal(0.01,0.0005,Epoch)
    model_name = 'generate12'
    neuron_size = [2]+[10]*10

    sigma_scale = bm.linspace(0.001, 1, len(neuron_size))
    tau = 1 / sigma_scale * 0.1

    # sigma_scale = bm.linspace(0.1, 0.1, len(neuron_size))
    # tau = np.linspace(1,1,len(neuron_size))*0.1

    duration = 100
    activation = bm.leaky_relu
    pcn = PCN(neuron_size, tau=tau, sigam_scale=sigma_scale, activation=activation, m = 0, noise = 2, zero_lastpar = True)

    batch_size = 100
    test_batch_size = 1000
    data_index = np.arange(data_length)
    if os.path.exists('./model/'+model_name+'_'+str(using_epoch)+'.bp'):
        states = bp.checkpoints.load_pytree('./model/'+model_name+'_'+str(using_epoch)+'.bp')
        pcn.load_state_dict(states)
    print(pcn.par_list)
    if train:
        for epoch in range(Epoch):
            p = 0
            np.random.shuffle(data_index)
            for i in range(0,data_length-batch_size,batch_size):
                data_choice = data_index[i:i+batch_size]
                input = bm.asarray(data[data_choice,:])
                x_time, e_time = sample(True, pcn, duration=duration, input=input)
                grad = pcn.learn(x_time, e_time, learning_rate[epoch])
                p += batch_size / data_length
                print(epoch, ':', int(p * 100), '%, grad=', grad)
                plot_neuron_2(x_time, e_time, 5, 5)
            bp.checkpoints.save_pytree('./model/' + model_name + '_' + str(epoch) + '.bp', pcn.state_dict())
            # pcn.save_states('./model/net.h5')


    x_0 = pcn.generate(test_batchsize=test_batch_size)
    # x_time, e_time = sample(False, pcn, batch_size=test_batch_size, duration=duration)
    # plot_neuron_2(x_time, e_time, 5, 5)
    # x_0 = x_time[0][-1,:,:]

    print(x_0.shape)
    plt.figure()
    plt.scatter(x_0[:,0],x_0[:,1])
    plt.show()