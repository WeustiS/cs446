import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torchvision.transforms.functional as TF
import random
import torchio as tio

def reshape(tensor):
    n = 1
    if len(tensor.shape) == 4:
        tensor = np.expand_dims(tensor, 0)

    n, c, x, y, z = tensor.shape

    if x>192:
        x = 192
        tensor = tensor[:,:,:192,:,:]
    if y>192:
        y = 192
        tensor = tensor[:,:,:,:192,:]
    if z>192:
        z = 192
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



def prep_dataset(dataset, names, device, train, augmenter):
    if train:
        d = 4
    else:
        d= 1
    xf, yf, zf = (192, 192, 192)
    
    ret = torch.empty((len(names), d, xf, yf, zf)).to('cpu')

    index = 0
    for i in names:

        sample =np.load("data_pub/"+i+'.npy')

        if len(sample.shape) == 3:
            sample = np.expand_dims(sample, 0)


        d, x, y, z = sample.shape
        if augmenter is not None:
            sample = augmenter(sample)
        sample = reshape(sample)
 
        sample = torch.from_numpy(sample).to('cpu')
 
        ret[index] = sample

        index += 1



    #print(f"Prepped Dataset of shape: {ret.shape}")
    return ret.to(device)


def get_batch(dataset, 	device):

    slice = sorted(list(dataset))[:-20]
    train_names = np.array([x for x in slice if x[-4:] == 'imgs'])

    samples = np.random.choice(len(train_names), len(train_names), replace=False)
    train = train_names[(samples)]
    test_names = np.array([x for x in slice if x[-3:] == 'seg'])
    test = test_names[(samples)]
    # augmenter = augmentClass()
    X = prep_dataset(dataset, train, device, True, None)

    y = prep_dataset(dataset, test, device, False, None)

    return X, y

def get_test(dataset, device):

    slice = sorted(list(dataset))[-20:]
    train_names = np.array([x for x in slice if x[-4:] == 'imgs'])

    samples = np.random.choice(len(train_names), len(train_names), replace=False)
    train = train_names[(samples)]
    test_names = np.array([x for x in slice if x[-3:] == 'seg'])
    test = test_names[(samples)]

    X = prep_dataset(dataset, train, device, True, None)

    y = prep_dataset(dataset, test, device, False, None)

    return X, y

class augmentClass():
    def __init__(self):
       self.affine = tio.RandomAffine(
           scales=(.85, 1.15),
           degrees=20,
           isotropic=True,
           image_interpolation='linear'
       )
       self.elastic = tio.RandomElasticDeformation(
           num_control_points=8, 
           max_displacement=4,
           locked_borders=2,
           image_interpolation='linear'
       )
       self.flip = tio.RandomFlip(axes=('LR'))
       
       scale = random.random()*.2+.9
       fliplr = bool(random.getrandbits(1))
       fliph = bool(random.getrandbits(1))
 
    def __call__(self, img):
        return self.affine(self.elastic(self.flip(img)))
 #   angle = random.randint(-20, 20)
 #   train = TF.rotate(train, angle)
 #   test = TF.rotate(test, angle)

 #   scale = random.random()*.2+.9
 #   train = TF.scale(train, scale)
 #   test = TF.scale(test, scale)

 #   fliplr = bool(random.getrandbits(1))
 #   if fliplr:
 #       train = TF.hflip(train)
 #       test = TF.hflip(test)

 #   fliph = bool(random.getrandbits(1))
 #   if fliph:
 #       train = TF.vflip(train)
  #      test = TF.vflip(test)

'''
fig, axs = plt.subplots(nrows=1, ncols=5,figsize=(12, 6))
ani = animation.FuncAnimation(axs[0], lambda i: plt.imshow(dataset['data_pub/train/001_imgs'][0, :, :, i]), np.arange(0, 150), init_func=lambda : plt.imshow(dataset['data_pub/train/001_imgs'][0, :, :, 0]), interval=20)
ani = animation.FuncAnimation(axs[0], lambda i: plt.imshow(dataset['data_pub/train/001_imgs'][1, :, :, i]), np.arange(0, 150), init_func=lambda : plt.imshow(dataset['data_pub/train/001_imgs'][1, :, :, 0]), interval=20)
ani = animation.FuncAnimation(axs[0], lambda i: plt.imshow(dataset['data_pub/train/001_imgs'][2, :, :, i]), np.arange(0, 150), init_func=lambda : plt.imshow(dataset['data_pub/train/001_imgs'][2, :, :, 0]), interval=20)
ani = animation.FuncAnimation(axs[0], lambda i: plt.imshow(dataset['data_pub/train/001_imgs'][3, :, :, i]), np.arange(0, 150), init_func=lambda : plt.imshow(dataset['data_pub/train/001_imgs'][3, :, :, 0]), interval=20)
ani = animation.FuncAnimation(axs[0], lambda i: plt.imshow(dataset['data_pub/train/001_seg'][:, :, i]), np.arange(0, 150), init_func=lambda : plt.imshow(dataset['data_pub/train/001_seg'][:, :, 0]), interval=20)
plt.show()

'''