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
from config import load_config
from PIL import Image


class PCN(bp.DynamicalSystemNS):
    def __init__(self, neuron_size, tau, sigam_scale, activation=bm.relu, name=None, batch_size=1, m=1, noise=2):
        super(PCN, self).__init__(name=name)
        # random_state
        self.rng = bm.random.RandomState()

        self.neuron_size = neuron_size
        self.layer = len(neuron_size)
        self.sigma_scale = sigam_scale
        self.activation = activation
        self.tau = tau
        self.par_num = 0
        self.m = m
        self.noise = noise

        # parameters

        self.par_list = bm.ListVar([])
        for i in range(self.layer - 1):
            self.par_list.append(self.rng.rand(neuron_size[i + 1], neuron_size[i]))
            self.par_num += neuron_size[i + 1] * neuron_size[i]
        self.par_list.append(self.rng.rand(1, neuron_size[-1]))
        self.par_num += neuron_size[-1]

        self.sigma_list = []
        for i in range(self.layer):
            self.sigma_list.append(bm.eye(self.neuron_size[i]) * self.sigma_scale[i])

        self.e = bm.ListVar([])
        for s in self.neuron_size:
            self.e.append(bm.zeros((batch_size, s)))

        self.var_list = bm.Variable(self.rng.normal(0, 1, size=(self.layer, batch_size, self.neuron_size[0])))
        self.v_list = bm.Variable(bm.zeros((self.layer, batch_size, self.neuron_size[0])))

        # self.var_list = bm.ListVar([])
        # self.v_list = bm.ListVar([])
        # for i in range(self.layer):
        #     self.var_list.append(self.rng.normal(0,1,size = (batch_size, self.neuron_size[i])))
        #     self.v_list.append(bm.zeros((batch_size, self.neuron_size[i])))

    def reset_neuron(self, batch_size=1, input=None):
        if input != None:
            batch_size = input.shape[0]
        for i, s in enumerate(self.neuron_size):
            self.e[i] = bm.zeros((batch_size, s))
        for i in range(self.layer):
            self.var_list[i] = self.rng.normal(0, 1, size=(batch_size, self.neuron_size[i]))
            self.v_list[i] = bm.zeros((batch_size, self.neuron_size[i]))
        if input != None:
            self.var_list[0] = input
        else:
            self.var_list[-1] = self.rng.normal(self.par_list[-1].squeeze(), self.sigma_scale[-1],
                                                size=(batch_size, self.neuron_size[-1]))
        # print(self.var_list[-1].shape)



    def update(self, train, time):
        batch_size = self.var_list[0].shape[0]
        var_list = [v.value for v in self.var_list]
        v_list = [v.value for v in self.v_list]
        es = [v.value for v in self.e]

        dt_global = bm.dt / 1000

        n = self.layer - 1



        def f1(var_pre, var_post, par, sigma):
            return (var_pre - self.activation(var_post) @ par) @ bm.linalg.inv(sigma)

        self.e[:self.layer-1] = jax.vmap(f1)(self.var_list[:self.layer-1],
                                             self.var_list[1:],
                                             self.par_list[:self.layer-1],
                                             self.sigma_list[:self.layer-1])
        self.e[-1] = ((self.var_list[-1] - self.par_list[-1]) @ bm.linalg.inv(self.sigma_list[-1]))






        for i in range(self.layer - 1):
            self.e[i].value = (
                    (self.var_list[i] - self.activation(self.var_list[i + 1]) @ self.par_list[i]) @ bm.linalg.inv(
                self.sigma_list[i]))

        for i in range(1, self.layer):
            dt = dt_global / self.tau[i]
            a = self.par_list[i - 1]  # (n_i,n_i-1)
            b = bm.vector_grad(self.activation)(self.var_list[i])  # (batch,n_i)
            c = jax.vmap(lambda b1: bm.expand_dims(b1, axis=1) * a)(b)  # (batch,n_i,n_i-1)
            tmp_noise = self.rng.normal(0, self.noise, (batch_size, self.neuron_size[i]))
            self.var_list[i].value = self.var_list[i] + (
                    -self.e[i] + bm.einsum('bjk,bk->bj', c, self.e[i - 1]) - self.v_list[
                i]) * dt + tmp_noise * bm.sqrt(dt)
            self.v_list[i].value = self.v_list[i] + (-self.m * self.v_list[i]) * dt + 2 * self.m * tmp_noise * bm.sqrt(
                dt)

        if not train:
            dt = dt_global / self.tau[0]
            tmp_noise = self.rng.normal(0, self.noise, (batch_size, self.neuron_size[0]))
            self.var_list[0].value = self.var_list[0] + (-self.e[0] - self.v_list[0]) * dt + tmp_noise * bm.sqrt(dt)
            self.v_list[0].value = self.v_list[0] + (-self.m * self.v_list[0]) * dt + 2 * self.m * tmp_noise * bm.sqrt(
                dt)

        # if train:
        #     for i in range(1, self.layer):
        #         dt = dt_global / self.tau[i]
        #         a = self.par_list[i - 1]  # (n_i,n_i-1)
        #         b = bm.vector_grad(self.activation)(self.var_list[i])  # (batch,n_i)
        #         c = jax.vmap(lambda b1: bm.expand_dims(b1, axis=1) * a)(b)  # (batch,n_i,n_i-1)
        #         tmp_noise = self.rng.normal(0, 2, (batch_size, self.neuron_size[i]))
        #         self.var_list[i].value = self.var_list[i] + (-self.e[i] + bm.einsum('bjk,bk->bj', c, self.e[i - 1]) - self.v_list[i]) * dt + tmp_noise * bm.sqrt(dt)
        #         self.v_list[i].value = self.v_list[i] + (-self.m * self.v_list[i]) * dt + 2 * self.m * tmp_noise * bm.sqrt(dt)
        #
        # if not train:
        #     for i in range(0, self.layer):
        #         dt = dt_global / self.tau[i]
        #         tmp_noise = self.rng.normal(0, 0, (batch_size, self.neuron_size[i]))
        #         self.var_list[i].value = self.var_list[i] + (-self.e[i]) * dt + tmp_noise * bm.sqrt(dt)
        #         #self.v_list[i].value = self.v_list[i] + (-self.m * self.v_list[i]) * dt

        return var_list, es

    def learn(self, x_time, e_time, rate):
        '''
        for i in range(self.layer - 1):
          #print(rate * bm.mean(bm.einsum('btn,btm->btnm', x_time[i + 1], e_time[i]), axis=(0, 1)))
          self.par_list[i].value = self.par_list[i] + rate * bm.mean(bm.einsum('btn,btm->btnm', x_time[i+1],e_time[i]),axis = (0,1))
        self.par_list[-1].value = self.par_list[-1] + rate * bm.mean(e_time[-1],axis = (0,1))
        '''
        sample_length = int(x_time[0].shape[0] / 100)
        grad = 0
        for i in range(self.layer - 1):
            # tmp_grad = bm.mean(bm.einsum('tbn,tbm->tbnm', self.activation(x_time[i + 1]), e_time[i])[-1,:,:,:],axis=0)
            tmp1 = self.activation(x_time[i + 1])[-sample_length:, :, :]
            tmp2 = e_time[i][-sample_length:, :, :]
            tmp_grad = bm.mean(bm.einsum('tbn,tbm->tbnm', tmp1, tmp2), axis=(0, 1))
            grad += bm.sum(bm.abs(tmp_grad))
            self.par_list[i].value = self.par_list[i] + rate * tmp_grad

        # tmp_grad = bm.mean(e_time[-1][-1,:,:],axis=0)
        tmp_grad = bm.mean(e_time[-1][-sample_length:, :, :], axis=(0, 1))
        grad += bm.sum(bm.abs(tmp_grad))
        self.par_list[-1].value = self.par_list[-1] + rate * tmp_grad

        grad /= self.par_num
        return grad

    def __save_state__(self) -> dict:
        r = {f'par_{i}': p for i, p in enumerate(self.par_list)}
        r['rng'] = self.rng
        for i, s in enumerate(self.sigma_list):
            r[f'sigma_{i}'] = s
        return r

    def __load_state__(self, state_dict: dict):
        for i, p in enumerate(self.par_list):
            p.value = state_dict[f'par_{i}']
        # self.rng.value = state_dict['rng']
        for i, s in enumerate(self.sigma_list):
            s.value = state_dict[f'sigma_{i}']
        return (), ()

    def generate(self, test_batchsize=10):
        # self.var_list[-1] = self.rng.normal(self.par_list[-1].squeeze(),self.sigma_scale[-1],size = (test_batchsize,self.neuron_size[-1]))
        self.var_list[-1] = self.rng.normal(self.par_list[-1].squeeze(), self.noise / 2,
                                            size=(test_batchsize, self.neuron_size[-1]))
        for i in range(self.layer - 1, 0, -1):
            self.var_list[i - 1] = self.activation(self.var_list[i]) @ self.par_list[i - 1] + self.rng.normal(0, 0, (
            test_batchsize, self.neuron_size[i - 1]))
        return self.var_list[0]


def sample(train, pcn, batch_size=1, duration=10, input=None):
    pcn.reset_neuron(batch_size=batch_size, input=input)
    times = bm.arange(0., duration, bm.dt)
    x_time, e_time = bm.for_loop(partial(pcn, train), times, child_objs=pcn)
    return x_time, e_time


def plot_neuron_1(neuron_size, e_time, x_time, sample_num):
    for i in range(len(neuron_size)):
        print(x_time[i][-1, 0, :])

    plt.figure()
    for i in range(len(neuron_size)):
        for j in range(sample_num):
            plt.subplot(len(neuron_size), sample_num, i * sample_num + j + 1)
            plt.plot(bm.arange(x_time[i].shape[0]), x_time[i][:, 0, j])
    plt.suptitle('neuron')

    plt.figure()
    for i in range(len(neuron_size)):
        for j in range(sample_num):
            plt.subplot(len(neuron_size), sample_num, i * sample_num + j + 1)
            plt.plot(bm.arange(e_time[i].shape[0]), e_time[i][:, 0, j])
    plt.suptitle('error')
    plt.show()


def plot_neuron_2(x_time, e_time, sample_num, batch_size):
    def plot_sub(x_time, name, sample_num):
        plt.figure()
        x_0 = x_time[0]
        for layer in range(len(x_time)):
            for i in range(sample_num):
                plt.subplot(len(x_time), sample_num, layer * sample_num + i + 1)
                for batch in range(min(x_0.shape[1], batch_size)):
                    plt.plot(bm.arange(x_time[layer].shape[0]), x_time[layer][:, batch, i], label=str(batch))
                    plt.ylabel('layer' + str(layer))
                    plt.xlabel('neuron' + str(i))
                if layer == 0 and i == 0:
                    plt.legend()

        plt.suptitle(name)
        plt.savefig('./result_image/test_' + name + '.png')
        # plt.show()
        plt.close()

    plot_sub(x_time, 'neuron ', sample_num)
    plot_sub(e_time, 'error ', sample_num)


def show_2D(x_0, image_shape, inv_normalize, average=False, rgb=False):
    for j in range(x_0.shape[1]):
        if average == False:
            output = x_0[-1, j, :].reshape(image_shape)
            output = inv_normalize(torch.from_numpy(bm.as_numpy(output)))
        else:
            sample_length = int(x_0.shape[0] / 10)
            output = bm.mean(x_0[-sample_length:, j, :], axis=0).reshape(image_shape)
            output = inv_normalize(torch.from_numpy(bm.as_numpy(output)))
        toPIL = transforms.ToPILImage()
        pic = toPIL(output.squeeze())
        base_width = 300
        h_size = int(float(pic.size[1]) * float(base_width / float(pic.size[0])))
        pic = pic.resize((base_width, h_size), Image.ANTIALIAS)
        pic.save('./result_image/sample' + str(j) + '.png')
        pic.close()
        # plt.show()


def show_video(x_0, image_shape, inv_normalize):
    import moviepy.editor as mp
    toPIL = transforms.ToPILImage()
    for batch in range(x_0.shape[1]):
        frames = []
        for i in range(x_0.shape[0]):
            output = x_0[i, batch, :].reshape(image_shape)
            output = inv_normalize(torch.from_numpy(bm.as_numpy(output)))
            pic = toPIL(output)
            base_width = 300
            h_size = int(float(pic.size[1]) * float(base_width / float(pic.size[0])))
            pic = pic.resize((base_width, h_size), Image.ANTIALIAS)
            frames.append(pic)
        frames[0].save('./result_image/video_' + str(batch) + '.gif', save_all=True, append_images=frames, loop=1,
                       duration=1)
        clip = mp.VideoFileClip('./result_image/video_' + str(batch) + '.gif')
        clip.write_videofile('./result_image/video_' + str(batch) + '.mp4')
    for batch in range(x_0.shape[1]):
        os.remove('./result_image/video_' + str(batch) + '.gif')


def cos_anneal(max, min, L):
    return min + 0.5 * (max - min) * (1 + bm.cos(bm.pi * bm.arange(L) / L))


if __name__ == '__main__':
    bm.set_platform('cpu')
    if torch.cuda.is_available():
        bm.set_platform('gpu')
        print('use gpu')
    else:
        print('use cpu')
    train = True

    training_batch_size = 16
    test_batch_size = 128

    # train_para
    Epoch = 10
    learning_rate = cos_anneal(0.01, 0.0005, Epoch)

    # pcn para
    neuron_size, model_name, normalize, inv_normalize, _transforms, training_data, using_epoch = load_config(4)

    sigma_scale = bm.linspace(0.01, 1, len(neuron_size))
    tau = 1 / sigma_scale
    duration = 100
    activation = bm.leaky_relu

    # load_data
    data_lenth = len(training_data)
    train_dataloader = DataLoader(training_data, batch_size=training_batch_size, shuffle=True)
    for input, label in train_dataloader:
        image_shape = input.shape[1:]
        image_size = 1
        for j in image_shape:
            image_size *= j
        break
    print('image_shape=', image_shape)
    print('image_size=', image_size)
    print('data_lenth=', data_lenth)

    # model
    neuron_size[0] = image_size
    pcn = PCN(neuron_size, tau=tau, sigam_scale=sigma_scale, activation=activation, m=5)

    if os.path.exists('./model/' + model_name + '_' + str(using_epoch) + '.bp'):
        states = bp.checkpoints.load_pytree('./model/' + model_name + '_' + str(using_epoch) + '.bp')
        pcn.load_state_dict(states)

    # training
    if train == True:
        for epoch in range(Epoch):
            p = 0
            for input, label in train_dataloader:
                input = bm.asarray(input.reshape(-1, neuron_size[0]))
                x_time, e_time = sample(True, pcn, duration=duration, input=input)
                grad = pcn.learn(x_time, e_time, learning_rate[epoch])

                p += training_batch_size / data_lenth
                print(epoch, ':', int(p * 100), '%, grad=', grad)

                # if int(p*100) == 0:
                #   plot_neuron_2(x_time, e_time, 5 , 5)

            # save states
            bp.checkpoints.save_pytree('./model/' + model_name + '_' + str(epoch) + '.bp', pcn.state_dict())
            # pcn.save_states('./model/net.h5')

    x_0 = pcn.generate(test_batchsize=test_batch_size)
    show_2D(bm.array([x_0]), image_shape, inv_normalize, average=False)
    print('x_0.shape=', x_0.shape)
    # show
    # x_time, e_time = sample(False, pcn, batch_size=test_batch_size, duration=duration)
    # print('running_finish')
    # plot_neuron_2(x_time, e_time, 10 , 5)
    # show_2D(x_time[0], image_shape, inv_normalize, average = False)
    # show_video(x_time[0], image_shape, inv_normalize)
