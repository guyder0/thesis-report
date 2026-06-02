path_to_data = 'data/voxel-man/'

import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import json

id_min, id_max = 1256, 2029
id_diff = id_max - id_min + 1
height, width = 330, 573

labels = np.ones((height, width, id_diff), dtype=np.uint32)

for i, id in enumerate(tqdm(range(id_min, id_max + 1))):
    img = Image.open(path_to_data + f'labels/labels{id}.tif')
    labels[..., i] = np.array(img)

unique_labels = np.unique(labels)

df = pd.read_excel(path_to_data + 'labels.xlsx', header=None).fillna(65000)
df.loc[:, 1:] = df.loc[:, 1:].astype(np.uint16)

labels_desc = {}
for label in unique_labels:
    try:
        row = df.loc[(df.loc[:, 1:].astype(np.uint16) == label).any(axis=1), df.columns[0]].iloc[0]
        labels_desc[int(label)] = row
    except Exception as e:
        ...

with open(path_to_data + 'labels.json', 'w') as f:
    f.write(json.dumps(labels_desc, indent=4))