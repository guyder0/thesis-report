import bpy
import random
import math
import struct

from mathutils import Vector, Matrix
from pathlib import Path


DATASET_PATH = Path('generated/')
dataset_path = Path(__file__).parent.parent / DATASET_PATH

images_path = dataset_path / 'images'
sparse_path = dataset_path / 'sparse/0'

images_file = sparse_path / 'images.bin'
cameras_file = sparse_path / 'cameras.bin'


def get_next_index():
    if not images_path.exists():
        images_path.mkdir(parents=True, exist_ok=True)
        return 1
    
    existing_files = list(images_path.glob("IMG_*.png"))
    if not existing_files:
        return 1

    indices = [int(f.stem.split('_')[1]) for f in existing_files]
    return max(indices) + 1


def save_cameras_bin(cam_ob):
    sparse_path.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    render = scene.render
    width = render.resolution_x
    height = render.resolution_y
    
    f_mm = cam_ob.data.lens
    sensor_w = cam_ob.data.sensor_width
    f_px = (f_mm / sensor_w) * width
    
    cx = width / 2
    cy = height / 2

    # num_cameras(Q), id(I), model(i), w(Q), h(Q), params(4d)
    with open(cameras_file, 'wb') as f:
        f.write(struct.pack("<Q", 1)) # 1 камера
        f.write(struct.pack("<IiQQ", 1, 1, width, height)) # ID=1, PINHOLE=1
        f.write(struct.pack("<dddd", f_px, f_px, cx, cy))


def update_images_bin(qvec, tvec, image_id, name):
    if not images_file.exists():
        with open(images_file, "wb") as f:
            f.write(struct.pack("<Q", 1)) # num_images = 1
            write_image_entry(f, qvec, tvec, image_id, name)
        return

    with open(images_file, "r+b") as f:
        old_count_bytes = f.read(8)
        old_count = struct.unpack("<Q", old_count_bytes)[0]
        new_count = old_count + 1
        
        f.seek(0)
        f.write(struct.pack("<Q", new_count))
        
        f.seek(0, 2)
        write_image_entry(f, qvec, tvec, image_id, name)


def write_image_entry(f, qvec, tvec, image_id, name):
    # IMAGE_ID (I), QVEC (4d), TVEC (3d), CAMERA_ID (I)
    f.write(struct.pack("<I", image_id))
    f.write(struct.pack("<dddd", *qvec))
    f.write(struct.pack("<ddd", *tvec))
    f.write(struct.pack("<I", 1))
    
    # NAME (строка + нулевой байт)
    f.write(name.encode("utf-8") + b"\0")
    
    # NUM_POINTS2D (Q)
    f.write(struct.pack("<Q", 0))
    

def get_colmap_pose(cam_ob):
    # Матрица перемещения из Blender (+Y вверх, -Z вперед) 
    # в COLMAP (+Y вниз, +Z вперед)
    flip_yz = Matrix.Diagonal((1, -1, -1, 1))
    
    matrix_w2c = (cam_ob.matrix_world @ flip_yz).inverted()
    
    quat = matrix_w2c.to_quaternion()
    tvec = matrix_w2c.translation
    return (quat.w, quat.x, quat.y, quat.z), (tvec.x, tvec.y, tvec.z)


def set_random_cam(cam_ob, dist):
    CENTER = Vector((0, 0, 0))
    
    cos_phi = random.uniform(-1, 1)
    sin_phi = (1 - cos_phi**2) ** 0.5
    theta = random.uniform(0, 2 * math.pi)

    pos = Vector((
        dist * sin_phi * math.cos(theta),
        dist * sin_phi * math.sin(theta),
        dist * cos_phi
    ))

    cam_ob.location = pos

    direction = (CENTER - pos).normalized()
    quat = direction.to_track_quat('-Z', 'Y')
    cam_ob.rotation_quaternion = quat
    

def main(num_images_to_generate):
    cam_ob = bpy.data.objects.get("Camera")
    cam_ob.rotation_mode = 'QUATERNION'
    
    start_idx = get_next_index()
    print(f"Начинаем с индекса: {start_idx}")
    
    if not cameras_file.exists():
        print(f'Создается файл {cameras_file}')
        save_cameras_bin(cam_ob)
    
    for i in range(num_images_to_generate):
        idx = start_idx + i
        img_name = f"IMG_{idx:04d}.png"
        
        set_random_cam(cam_ob, 100)
        bpy.context.view_layer.update()
        
        bpy.context.scene.render.filepath = str(images_path / img_name)
        bpy.ops.render.render(write_still=True)
        
        qvec, tvec = get_colmap_pose(cam_ob)
        
        update_images_bin(qvec, tvec, idx, img_name)
        
        print(f'({i+1} / {num_images_to_generate}) Успешно создан кадр')
        
    print(f"Генерация завершена. Создано кадров: {num_images_to_generate}")

if __name__ == '__main__':
    main(100)