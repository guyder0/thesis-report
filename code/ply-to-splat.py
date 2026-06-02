import numpy as np
from plyfile import PlyData

def process_ply_to_splat(ply_file_path, splat_file_path):
    print("Reading PLY...")
    plydata = PlyData.read(ply_file_path)
    v = plydata["vertex"]
    num_verts = v.count

    print("Calculating importance...")
    importance = np.exp(v["scale_0"] + v["scale_1"] + v["scale_2"]) / (1 + np.exp(-v["opacity"]))
    indices = np.argsort(-importance)

    dtype = [
        ('position', '<f4', 3),
        ('scales', '<f4', 3),
        ('color', '<u1', 4),
        ('rotation', '<u1', 4),
        ('segmentation', '<u4')
    ]
    
    splat_data = np.empty(num_verts, dtype=dtype)

    print("Filling data...")
    
    # Позиции
    splat_data['position'][:, 0] = v['x']
    splat_data['position'][:, 1] = v['y']
    splat_data['position'][:, 2] = v['z']

    # Масштабы (экспоненцированные)
    splat_data['scales'][:, 0] = np.exp(v['scale_0'])
    splat_data['scales'][:, 1] = np.exp(v['scale_1'])
    splat_data['scales'][:, 2] = np.exp(v['scale_2'])

    # Цвета (SH -> RGB + Opacity)
    SH_C0 = 0.28209479177387814
    r = (0.5 + SH_C0 * v['f_dc_0']) * 255
    g = (0.5 + SH_C0 * v['f_dc_1']) * 255
    b = (0.5 + SH_C0 * v['f_dc_2']) * 255
    a = (1 / (1 + np.exp(-v['opacity']))) * 255
    
    splat_data['color'][:, 0] = np.clip(r, 0, 255)
    splat_data['color'][:, 1] = np.clip(g, 0, 255)
    splat_data['color'][:, 2] = np.clip(b, 0, 255)
    splat_data['color'][:, 3] = np.clip(a, 0, 255)

    # Вращение (Нормализация кватернионов)
    rots = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1)
    qlen = np.linalg.norm(rots, axis=-1, keepdims=True)
    # Обрабатываем деление на ноль, если вдруг есть пустые кватернионы
    qlen[qlen == 0] = 1.0
    rots_norm = (rots / qlen) * 128 + 128
    
    splat_data['rotation'][:, 0] = np.clip(rots_norm[:, 0], 0, 255)
    splat_data['rotation'][:, 1] = np.clip(rots_norm[:, 1], 0, 255)
    splat_data['rotation'][:, 2] = np.clip(rots_norm[:, 2], 0, 255)
    splat_data['rotation'][:, 3] = np.clip(rots_norm[:, 3], 0, 255)

    # Сегментация
    splat_data['segmentation'] = v['segmentation'].astype(np.uint32)
    return splat_data['segmentation']

    print("Sorting...")
    splat_data = splat_data[indices]

    print(f"Saving to {splat_file_path}...")
    with open(splat_file_path, "wb") as f:
        f.write(splat_data.tobytes())
    
    print("Done!")
