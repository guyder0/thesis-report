from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

pth_ply = Path('results/voxel-man-default-no-normalize/ply')
pth_labels = Path('data/numpy/labels.npy')

def get_ply_data(file_path):
    ply = PlyData.read(file_path)
    # гауссианы записываются как vertex
    el = ply['vertex']
    
    field_names = [prop.name for prop in el.properties]
    # все параметры гауссиан по стандарту записаны как f4
    data = np.stack([el[name] for name in field_names], axis=1).astype(np.float32)
    
    return data, field_names


def save_ply_with_segmentation(file_path, points, fields, seg_data):
    dtype = [(name, 'f4') for name in fields]
    dtype.append(('segmentation', 'u4'))
    
    n_points = points.shape[0]
    seg_points = np.empty(n_points, dtype=dtype)
    
    for i, name in enumerate(fields):
        seg_points[name] = points[:, i]
    
    seg_points['segmentation'] = seg_data
    
    el = PlyElement.describe(seg_points, 'vertex')
    PlyData([el], text=False).write(file_path)


# согласованные ранее метки костной системы из

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


def transform_3d(
    labels, num_choice=100_000,
    # дефолтные преобразования сразу подставлены согласовано со скриптом 
    # processing_scripts/numpy_to_points3Dbin.py
    # кроме translate который вычисляется относительно
    s=np.array([0.1, 0.1, 0.1]), r=np.array([180, 0, 0]),
):
    '''
        input: labels (X, Y, Z)
        output: coords (N, 3) и labels (N,)
    '''
    t = np.array([-0.5, 0.5, 0.5]) * labels.shape * s
    
    mask = np.isin(labels, skeletal_system_ids)
    coords = np.argwhere(mask)
    labels = labels[mask]

    idx = np.random.choice(coords.shape[0], num_choice, replace=False)
    coords = coords[idx] # (N, 3)
    labels = labels[idx]
    
    xyz = np.vstack((coords.T, np.ones(num_choice))) # (4, N)
    
    S = np.diag([*s, 1])

    Rx, Ry, Rz = np.eye(4), np.eye(4), np.eye(4)
    theta_x, theta_y, theta_z = np.radians(r)
    
    Rx[:3, :3] = np.array([
        [1, 0,                0              ],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x),  np.cos(theta_x)],
    ])
    
    Ry[:3, :3] = np.array([
        [ np.cos(theta_y), 0, np.sin(theta_y)],
        [ 0,               1, 0              ],
        [-np.sin(theta_y), 0, np.cos(theta_y)],
    ])

    Rz[:3, :3] = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z),  np.cos(theta_z), 0],
        [0,                0,               1],
    ])

    R = Rx @ Ry @ Rz
    
    T = np.eye(4)
    T[:3, 3] = t

    M = T @ R @ S
    xyz = M @ xyz # (4, N)
    coords = xyz[:3, :].T # (N, 3)
    
    return coords.astype(np.float32), labels


inspected_ply = pth_ply / 'point_cloud_29999.ply'
segmented_ply = pth_ply / 'segmented.ply'

# загружает обученные гауссианы
inspect_ply(inspected_ply)
points, fields = get_ply_data(inspected_ply)

# загружаем 1_000_000 случайных меток и трансформируем их в соответствии с colmap
labels = np.load(pth_labels)
coords, labels = transform_3d(labels, 1_000_000)


def transfer_segmentation_kdtree(target_points, source_coords, source_labels, k=1, workers=-1):
    tree = cKDTree(source_coords)
    distances, indices = tree.query(target_points, k=1, workers=workers)
    target_labels = source_labels[indices]

    # визуализация distances чтобы быть уверенным что всё нормально приклеелось
    # plt.figure(figsize=(10, 6))
    
    # n, bins, patches = plt.hist(distances, bins=100, color='skyblue', edgecolor='black', log=True)
    # plt.axvline(np.mean(distances), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(distances):.2f}')
    # plt.axvline(np.median(distances), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(distances):.2f}')
    
    # plt.title('Распределение расстояний от Гауссиан до ближайшего вокселя')
    # plt.xlabel('Евклидово расстояние')
    # plt.ylabel('Количество точек (логарифмическая шкала)')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    return target_labels


target_labels = transfer_segmentation_kdtree(points[:, :3], coords, labels)

save_ply_with_segmentation(
    segmented_ply, 
    points, 
    fields, 
    target_labels,
)