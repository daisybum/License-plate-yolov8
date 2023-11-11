import os
import json
import pickle
from glob import glob

import pandas as pd
from tqdm import tqdm

data_path = "D:\\number_plate_dataset\\datasets_unicode"
json_paths = os.path.join(data_path, "vehicle_type\\*\\*")

#
df = pd.read_parquet('../data/labels_v1.parquet')

codes_train = pd.read_parquet('../data/plate_ids_train.parquet')
codes_valid = pd.read_parquet('../data/plate_ids_train.parquet')

# Using tqdm for progress indication
for row in tqdm(df.itertuples()):
    if len(codes_train[codes_train['id'].str.contains(row.id)]):
        df.loc[row.Index, 'image_name'] = codes_train[codes_train['id'] == row.id]['image_path']
    else:
        continue

# with open('..\\data\\img_path_dict.pkl', 'rb') as f:
#     img_bbox_dict = pickle.load(f)
#
# bbox_list = [bbox[0] for bbox in list(img_bbox_dict.values())]
#
# nan_indice = []
# for idx, val in enumerate(bbox_list):
#     try:
#         len(val)
#     except TypeError:
#         nan_indice.append(idx)
