import gc
import numpy as np
import pyopenvdb as vdb

def convert_npy_to_vdb(rgb_dir, ct_dir, output_dir):
    rgb = np.load(rgb_dir, mmap_mode='r')
    ct = np.load(ct_dir, mmap_mode='r')

    grid_rgb = vdb.Vec3SGrid()
    grid_rgb.copyFromArray(rgb)
    grid_rgb.name = 'rgb'

    del rgb
    gc.collect()
    print('RGB скопирован, память очищена')
    
    grid_ct = vdb.FloatGrid()
    grid_ct.copyFromArray(ct)
    grid_ct.name = 'ct'

    del ct
    gc.collect()
    print('CT скопирован, память очищена')

    vdb.write(output_dir, [grid_ct, grid_rgb])
    print(f"Готово! Файл {output_dir} создан.")

def main(
    rgb_dir: str,
    ct_dir: str,
    output_dir: str
) -> None:
    convert_npy_to_vdb(rgb_dir, ct_dir, output_dir)

if __name__ == '__main__':
    import tyro
    tyro.cli(main)