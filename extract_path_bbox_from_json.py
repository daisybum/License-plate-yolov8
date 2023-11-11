import os
import json
import pickle
from glob import glob

import pandas as pd
from tqdm import tqdm

data_path = "D:\\number_plate_dataset\\datasets_unicode"
json_paths = glob(os.path.join(data_path, "vehicle_type\\*\\*\\*.json"))

#
df = pd.DataFrame(
    columns=
    [
        'image_path', 'car_x1', 'car_y1', 'car_x2', 'car_y2', 'plate_x1', 'plate_y1', 'plate_x2', 'plate_y2', 'id'
    ])

# Using tqdm for progress indication
for index, json_path in tqdm(enumerate(json_paths[479999:])):
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        data_dict = json.load(f)

    # Extracting the imagePath and plate bbox
    try:
        df.loc[index] = {'image_path': data_dict['imagePath'],
                         'car_x1': data_dict['car']['bbox'][0][0],
                         'car_y1': data_dict['car']['bbox'][0][1],
                         'car_x2': data_dict['car']['bbox'][1][0],
                         'car_y2': data_dict['car']['bbox'][1][1],
                         'plate_x1': data_dict['plate']['bbox'][0][0],
                         'plate_y1': data_dict['plate']['bbox'][0][1],
                         'plate_x2': data_dict['plate']['bbox'][1][0],
                         'plate_y2': data_dict['plate']['bbox'][1][1],
                         'id': data_dict['id']
                         }
    except KeyError:
        continue

    if index % 10000 == 0:
        df.to_parquet("data\\labels_v1.parquet")

df.to_parquet("data\\labels.parquet")

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
