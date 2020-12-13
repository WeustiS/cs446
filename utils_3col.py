import os
import numpy as np

from glob import glob
import pandas as pd
from tqdm import tqdm

# example: BraTS20_Training_010_imgs.npy

data_dir = '/home/brianc5/cs446fa20/brain/validation' #None # path to data/npy folder

# from: https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66
def rlencode(x, dropna=False):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle 
    function from R.
    
    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    dropna: bool, optional
        Drop all runs of NaNs.
    
    Returns
    -------
    start positions, run lengths, run values
    
    """
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
                return (np.array([], dtype=int), 
                        np.array([], dtype=int), 
                        np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    
    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]
    
    return starts, lengths, values



def main():
    files = glob('{}/*'.format(data_dir))

    ids = []
    codes = []
    classId = []
    for i, file in enumerate(tqdm(files)):
        if not file.split('/')[-1][4:4+3] == 'seg':
            continue

        im_name = file.split('/')[-1][:3]
        sample = np.load(file).flatten()
        
        starts, lengths, values = rlencode(sample)
        code_1 = ''
        code_2 = ''
        code_3 = ''
        code_4 = ''
        for start, length, value in zip(starts, lengths, values):
            if value == 0:
                code_1 += '{} {} '.format(start, length)
            elif value == 1:
                code_2 += '{} {} '.format(start, length)
            elif value == 2:
                code_3 += '{} {} '.format(start, length)
            elif value == 3:
                code_4 += '{} {} '.format(start, length)

        # update final output list
        ids.extend([im_name] * 4)
        classId.extend(['0', '1', '2', '3'])
        codes.extend([code_1.strip(), code_2.strip(), code_3.strip(), code_4.strip()])


    dic = {'Id': ids, 'Expected': codes, 'ClassId': classId}
    df = pd.DataFrame(dic)
    df.to_csv('solution_val_3col.csv', index=False)

if __name__ == '__main__':
    main()





