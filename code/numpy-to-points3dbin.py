import numpy as np
import struct
import os
import gc
from tqdm import tqdm

def transform_3d(x, y, z, s=np.array([1, 1, 1]), r=np.array([0, 0, 0]), t=np.array([0, 0, 0])):
    num_points = x.shape[0]
    xyz = np.stack((x, y, z, np.ones(num_points)), axis=0) # (4, N)
    print(f'xyz {xyz.shape}')
    
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
    print(f'xyz {xyz.shape}')
    return xyz[:3, :] # (3, N)
    

def save_voxel_grid_to_colmap(rgb_grid, ct_grid, output_dir, num_choice=100_000):
    """
    rgb_grid: np.array формы (X, Y, Z, 3) - float32
    ct_grid: np.array формы (X, Y, Z) - float32
    output_dir: папка, куда сохранить файлы модели (points3D.bin)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mask = ct_grid > 0
    x_idx, y_idx, z_idx = np.where(mask)
    
    del ct_grid
    gc.collect()
    
    rgb_values = (rgb_grid[x_idx, y_idx, z_idx] * 255).astype(np.uint8)
    grid_shape = rgb_grid.shape[:3]
    
    del rgb_grid
    gc.collect()

    num_points = x_idx.shape[0]
    print(f"Найдено занятых вокселей: {num_points}")

    # Scale, Rotation, Transform (X, Y, Z)
    s = np.array([0.1, 0.1, 0.1])
    r = np.array([180, 0, 0])
    t = np.array([-0.5, 0.5, 0.5]) * grid_shape * s

    xyz_coords = transform_3d(x_idx, y_idx, z_idx, s, r, t)
    print('Координаты преобразованы')

    del x_idx, y_idx, z_idx
    gc.collect()

    points_path = os.path.join(output_dir, "points3D.bin")
    with open(points_path, "wb") as f:
        f.write(struct.pack("<Q", num_choice))
        choice = np.random.choice(np.arange(num_points), num_choice)
        
        for i, idx in zip(tqdm(range(num_choice)), choice):
            data = struct.pack(
                "<QdddBBBdQ",
                i + 1,                         # Point ID
                xyz_coords[0, idx],              # X
                xyz_coords[1, idx],              # Y
                xyz_coords[2, idx],              # Z
                rgb_values[idx, 0],              # R
                rgb_values[idx, 1],              # G
                rgb_values[idx, 2],              # B
                0.0,                           # Error
                0                              # Track Length
            )
            f.write(data)

    print(f"Готово! Модель сохранена в {output_dir}")


def main(
    rgb_dir: str,
    ct_dir: str,
    output_dir: str,
    num_choice: int = 100_000, # количество точек, случайно извлеченных из воксельной сетки
) -> None:
    
    rgb_grid = np.load(rgb_dir)
    print(f'Загружен RGB, dtype={rgb_grid.dtype}')
    ct_grid = np.load(ct_dir)
    print(f'Загружен CT, dtype={ct_grid.dtype}')
    
    save_voxel_grid_to_colmap(rgb_grid, ct_grid, output_dir, num_choice)


if __name__ == '__main__':
    import tyro
    tyro.cli(main)