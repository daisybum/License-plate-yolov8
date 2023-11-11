import os
import json
import pickle
from glob import glob

import pandas as pd
from tqdm import tqdm

data_path = ("D:\\Validation\\라벨링데이터\\차종분류데이터")
json_paths = glob(os.path.join(data_path, "*\\*\\*.json"))

#
df = pd.DataFrame(columns=['image_path', 'id'])

# Using tqdm for progress indication
for index, json_path in tqdm(enumerate(json_paths)):
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        data_dict = json.load(f)

    # Extracting the imagePath and plate bbox
    try:
        df.loc[index] = {'image_path': data_dict['car']['imagePath'],
                         'id': data_dict['id']
                         }
    except KeyError:
        continue

    if index % 10000 == 0:
        df.to_parquet("data\\plate_ids_valid.parquet")

df.to_parquet("data\\plate_ids_valid.parquet")

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

