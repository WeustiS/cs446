import numpy as np
import matplotlib.pyplot as plt
import torch


def reshape(tensor):
    n = 1
    if len(tensor.shape) == 4:
        tensor = np.expand_dims(tensor, 0)

    n, c, x, y, z = tensor.shape
    return np.pad(tensor, ((0,0), # n
                           (0,0), # c
                           (0, 192-x), # x
                           (0, 192-y), # y
                           (0, 192-z) # z
                           ),
                  mode='edge')

def view(x, y, slice):
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=5,
                                             figsize=(12, 6))
    ax0.imshow(x[0,:,:,slice])

    ax1.imshow(x[1,:,:,slice])

    ax2.imshow(x[2,:,:,slice])

    ax3.imshow(x[3, :, :,slice])

    ax3.imshow(y[:,:,slice])

    plt.show()



def prep_dataset(data, names, device, train):
    if train:
        d = 4
    else:
        d= 1
    xf, yf, zf = (192, 192, 192)

    ret = torch.empty((len(names), d, xf, yf, zf)).to(device)
    index = 0
    for i in names:
        try:
            print(i)
            sample = np.load(i)
            d, x, y, z = sample.shape

            sample = reshape(sample)
            sample = torch.from_numpy(sample).to(device)
            ret[index] = sample
            index += 1
        except:
            pass
    print(f"Prepped Dataset of shape: {ret.shape}")
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



