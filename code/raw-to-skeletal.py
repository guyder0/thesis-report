path_to_data = 'data/voxel-man/'

import numpy as np
from PIL import Image
from tqdm import tqdm

id_min, id_max = 1256, 2029
id_diff = id_max - id_min + 1
height, width = 330, 573

labels = np.ones((height, width, id_diff), dtype=np.uint32)
rgb = np.ones((height, width, id_diff, 3), dtype=np.uint8)
ct = np.ones((height, width, id_diff), dtype=np.uint8)

for i, id in enumerate(tqdm(range(id_min, id_max + 1))):
    img = Image.open(path_to_data + f'labels/labels{id}.tif')
    labels[..., i] = np.array(img)
    img = Image.open(path_to_data + f'rgb/rgb{id}.tif')
    rgb[:, :, i, :] = np.array(img)
    img = Image.open(path_to_data + f'CT/frozenCT{id}.tif')
    ct[:, :, i] = np.array(img)

filt_labels = ((labels != 0) & (labels // 100 != 73)) * labels
rgb = (rgb / 256).astype(np.float32)
ct = (ct / 256).astype(np.float32)

# Выделение интересующих классов

bones_ids = [
    # General
    6501,

    # Clavicles and sternum
    6502, 6503, 6504,

    # Cervical vertebrae
    6511, 6512, 6513,

    # Thoracic vertebrae
    6514, 6515, 6516, 6517, 6518,
    6519, 6520, 6521, 6522, 6523,
    6524, 6554,

    # Ribs
    6525, 6526,
    6527, 6528,
    6529, 6530,
    6531, 6532,
    6533, 6534,
    6535, 6536,
    6537, 6538,
    6539, 6540,
    6541, 6542,
    6543, 6544,
    6550, 6551,
    6552, 6553,

    # Upper limbs
    6546, 6547,   # humerus
    6566, 6567,   # right radius/ulna
    6568, 6569,   # left radius/ulna
    6570, 6571,   # hand bones

    # Scapulae
    6548, 6549,

    # Lumbar spine
    6555, 6556, 6557, 6558, 6559,

    # Pelvis and lower spine
    6560, 6561,
    6562, 6563,

    # Femurs
    6564, 6565
]

cartilage_ids = [
    7013,
    7036,
    7037, 7038, 7039, 7040, 7041, 7042,
    7044,
    7045, 7046, 7047, 7048, 7049, 7050
]

intervertebral_disc_ids = [
    7015, 7016, 7017, 7018, 7019,
    7020, 7021, 7022, 7023, 7024,
    7025, 7026, 7027, 7028, 7029,
    7030, 7031, 7032, 7033, 7034
]

skeletal_system_ids = (
    bones_ids +
    cartilage_ids +
    intervertebral_disc_ids
)

mask = np.isin(filt_labels, skeletal_system_ids).astype(np.uint8)

skeletal_rgb = rgb * mask[..., np.newaxis]
skeletal_ct = ct * mask

np.save('data/numpy/rgb_skeletal.npy', np.ascontiguousarray(skeletal_rgb))
np.save('data/numpy/ct_skeletal.npy', np.ascontiguousarray(skeletal_ct))