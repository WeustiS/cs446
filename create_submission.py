import os
import argparse
import numpy as np
import pandas as pd

from glob import glob
from tqdm import tqdm

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

def main(args):
    # get all the npy files in folder
    files = glob('{}/*'.format(args.seg_dir))

    
    ids = []    # ImageId_ClassId columns
    codes = []  # EncodedPixels columns

    # iterate through the entire folders. Compute the run-length encoding for each segmenation npy file
    for i, file in enumerate(tqdm(files)):
        # check if the file is seg (exclude img npy file)
        # npy file name are expected to have the format of XXX_seg.npy
        if not file.split('/')[-1][-7:-4] == 'seg':
            continue

        # im_name are expected to have the format of XXX_seg.npy
        im_name = file.split('/')[-1][:3]
        sample = np.load(file).flatten() # flatten the numpy array to 1d
        
        # get the start position, 
        # length for each segment correspoding to start position, 
        # and the value/labels for each segments
        starts, lengths, values = rlencode(sample)

        # these will be the encoded pixels strings for each image, 4 labels for each iamge
        code_1 = ''
        code_2 = ''
        code_3 = ''
        code_4 = ''
        # convert the run-length encoding output from rlencode function to the required csv format
        for start, length, value in zip(starts, lengths, values):
            if value == 0:
                code_1 += '{} {} '.format(start, length)
            elif value == 1:
                code_2 += '{} {} '.format(start, length)
            elif value == 2:
                code_3 += '{} {} '.format(start, length)
            elif value == 3:
                code_4 += '{} {} '.format(start, length)

        # update final output list, add for rows for each image (4 labels)
        for i in range(4):
            id = '{}_{}'.format(im_name, i)
            ids.append(id)
        codes.extend([code_1.strip(), code_2.strip(), code_3.strip(), code_4.strip()])


    dic = {'ImageId_ClassId': ids, 'EncodedPixels': codes}
    df = pd.DataFrame(dic)
    df.to_csv(args.out_csv_path, index=False)



# This is a sample script for outputing a run-legnth encoding csv file, which is required for kaggle to work
# Assume all predicted segmentation in npy format are stored in a folder
# this script output the run-length encoding csv file to the specified location

# Example output:

# ClassId_,Expected
# dummy1_0,0 1 2 2 6 2
# dummy1_1,1 1 4 2
# dummy2_0,2 2 5 1
# dummy2_1,0 2 4 1 6 2

# two dummy image each with label 0 and 1, whice gives us a total of 4 rows


# example command to use this scrip:
# python create_submission.py --seg-dir validation --out-csv-path temp.csv
# validation is the folder to your seg files, temp.csv is the output csv file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg-dir', type=str, default=None, required=True,
                        help='relative path to the data folder containg the segmentation npy file')
    parser.add_argument('--out-csv-path', type=str, default='submission.csv',
                        help='relative path to your output csv file')
    args = parser.parse_args()

    main(args)
