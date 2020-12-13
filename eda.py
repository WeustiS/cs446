import numpy as np
import matplotlib.pyplot as plt
import torch
import time

def reshape(tensor):
    n = 1
    if len(tensor.shape) == 4:
        tensor = np.expand_dims(tensor, 0)

    n, c, x, y, z = tensor.shape
    if x>192:
        tensor = tensor[:,:,:192,:,:]
    if y>192:
        tensor = tensor[:,:,:,:192,:]
    if z>192:
        tensor = tensor[:,:,:,:,:192]
    return np.pad(tensor, ((0,0), # n
                           (0,0), # c
                           (0, 192-x), # x
                           (0, 192-y), # y
                           (0, 192-z) # z
                           ),
                  mode='constant',
                  constant_values=0
                  )

def view(x, y, slice):
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=5,
                                             figsize=(12, 6))
    ax0.imshow(x[0,:,:,slice])

    ax1.imshow(x[1,:,:,slice])

    ax2.imshow(x[2,:,:,slice])

    ax3.imshow(x[3, :, :,slice])

    ax3.imshow(y[:,:,slice])

    plt.show()



def prep_dataset(dataset, names, device, train):
    if train:
        d = 4
    else:
        d= 1
    xf, yf, zf = (192, 192, 192)

    ret = torch.empty((len(names), d, xf, yf, zf)).to(device)
    index = 0
    for i in names:
        sample = np.load(i+'.npy')

        if len(sample.shape) == 3:
            sample = np.expand_dims(sample, 0)


        d, x, y, z = sample.shape


        sample = reshape(sample)
        sample = torch.from_numpy(sample).to(device)
        ret[index] = sample
        index += 1



    #print(f"Prepped Dataset of shape: {ret.shape}")
    return ret


def get_batch(dataset, batch_size, device):

    train_names = np.array(list(dataset)[1:409] + list(dataset)[410:])[::2]
    samples = np.random.choice(len(train_names), batch_size, replace=False)
    train = train_names[(samples)]
    test_names = np.array(list(dataset)[2:409] + list(dataset)[410:])[::2]
    test = test_names[(samples)]

    X = prep_dataset(dataset, train, device, True)
    y = prep_dataset(dataset, test, device, False)

    return X, y



'''
fig, axs = plt.subplots(nrows=1, ncols=5,figsize=(12, 6))
ani = animation.FuncAnimation(axs[0], lambda i: plt.imshow(dataset['data_pub/train/001_imgs'][0, :, :, i]), np.arange(0, 150), init_func=lambda : plt.imshow(dataset['data_pub/train/001_imgs'][0, :, :, 0]), interval=20)
ani = animation.FuncAnimation(axs[0], lambda i: plt.imshow(dataset['data_pub/train/001_imgs'][1, :, :, i]), np.arange(0, 150), init_func=lambda : plt.imshow(dataset['data_pub/train/001_imgs'][1, :, :, 0]), interval=20)
ani = animation.FuncAnimation(axs[0], lambda i: plt.imshow(dataset['data_pub/train/001_imgs'][2, :, :, i]), np.arange(0, 150), init_func=lambda : plt.imshow(dataset['data_pub/train/001_imgs'][2, :, :, 0]), interval=20)
ani = animation.FuncAnimation(axs[0], lambda i: plt.imshow(dataset['data_pub/train/001_imgs'][3, :, :, i]), np.arange(0, 150), init_func=lambda : plt.imshow(dataset['data_pub/train/001_imgs'][3, :, :, 0]), interval=20)
ani = animation.FuncAnimation(axs[0], lambda i: plt.imshow(dataset['data_pub/train/001_seg'][:, :, i]), np.arange(0, 150), init_func=lambda : plt.imshow(dataset['data_pub/train/001_seg'][:, :, 0]), interval=20)
plt.show()

'''