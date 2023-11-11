import os
import re
from glob import glob

from tqdm import tqdm
from unidecode import unidecode


def make_computer_friendly(file_name):
    """Function to convert file names"""
    # Remove or replace Korean characters with their Unidecode equivalents
    new_name = unidecode(file_name)
    # Replace spaces with underscores if needed
    new_name = new_name.replace(' ', '_')
    return new_name


if __name__ == "__main__":
    # Directory where files are located (change to your directory)
    directory = "D:\\number_plate_dataset\\plate_dataset\\datasets"
    file_paths = glob(os.path.join(directory, "*\\images\\*"))

    # Loop through all files in the directory
    for imagepath in tqdm(file_paths):
        labelpath = imagepath.replace('images', 'labels').replace('.jpg', '.txt')
        new_imagepath = make_computer_friendly(imagepath)
        new_labelpath = new_imagepath.replace('images', 'labels').replace('.jpg', '.txt')

        os.rename(imagepath, new_imagepath)
        os.rename(labelpath, new_labelpath)
        print(f'Renamed "{imagepath}" to "{new_labelpath}"')

    print('All files have been renamed.')
