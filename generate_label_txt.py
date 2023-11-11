import os
import pandas as pd
from tqdm import tqdm

DATA_OUT_DIR = "D:\\images\\val"
df_orig = pd.read_parquet(os.path.join(os.getcwd(), 'data\\labels.parquet'))
df_target = pd.read_parquet(os.path.join(os.getcwd(), 'data\\plate_ids_valid.parquet'))
df = pd.merge(df_target, df_orig, how='inner', on='id')

for row in tqdm(df.itertuples()):
    full_path = os.path.join(DATA_OUT_DIR, os.path.basename(row.image_path_x))
    image_name = os.path.basename(full_path)
    try:
        label_path = os.path.splitext(full_path)[0].replace('images', 'labels') + ".txt"
        with open(label_path, 'w') as f_ann:
            # class_id, xc, yx, w, h
            car_x1, car_x2, car_y1, car_y2 = [
                float(j) for j in [row.car_x1, row.car_x2, row.car_y1, row.car_y2]
            ]
            plate_x1, plate_x2, plate_y1, plate_y2 = [
                float(j) for j in [row.plate_x1, row.plate_x2, row.plate_y1, row.plate_y2]
            ]

            x1 = plate_x1 - car_x1
            y1 = plate_y1 - car_y1
            x2 = plate_x2 - car_x1
            y2 = plate_y2 - car_y1

            width = car_x2 - car_x1
            height = car_y2 - car_y1

            xc = (x1 + x2) / 2 / width
            yc = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height

            if h > 1:
                h = 0.999999

            f_ann.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
            f_ann.close()
    except FileNotFoundError:
        print(image_name + " not found")
        continue
