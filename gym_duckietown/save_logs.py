"""
Save logs is used to save logs experienced in the duckietown gym simulator.
Stored are:
    - The input Image (# Camera image size WIDTH = 160, HEIGHT = 120)
    - The received reward signal (if any)
    - The output commands
    - The pose and velocities of the Duckiebot
    - The ideal rule-based trajectory
"""
import os
import h5py
import numpy as np

PATH = 'data.h5'
TAGS = ['Images', 'Reward', 'Output', 'Position',
        'Angle', 'Velocity', 'Ref-Position']
INITIAL_SIZE = 100

#TODO: Print out this data against
#TODO: Is int8 enough for the image?


class SaveLogs():
    """
    Saves logs in h5 format in file data.h5
    """
    def __init__(self):
        if not os.path.exists(PATH):
            f = h5py.File(PATH, "w")
            f.close()
            os.remove(PATH)
            with h5py.File(PATH, "a") as f:
                dt = h5py.special_dtype(vlen=np.dtype('int8'))
                dset_img = f.create_dataset('Images', (INITIAL_SIZE, 120, 160),
                                            maxshape=(None, 120, 160), dtype=dt)
                dset_r = f.create_dataset('Reward', (INITIAL_SIZE,), maxshape=(None,),
                                        dtype='f')
                dset_o = f.create_dataset('Output', (INITIAL_SIZE, 2), maxshape=(None, 2),
                                        dtype='f')
                dset_p = f.create_dataset('Position', (INITIAL_SIZE, 3), maxshape=(None, 3),
                                        dtype='f')
                dset_a = f.create_dataset('Angle', (INITIAL_SIZE,), maxshape=(None,),
                                        dtype='f')
                dset_v = f.create_dataset('Velocity', (INITIAL_SIZE,), maxshape=(None,),
                                        dtype='f')
                dset_ref = f.create_dataset('Ref-Position', (INITIAL_SIZE, 3),
                                            maxshape=(None, 3), dtype='f')

    def add(self, img, reward, output, position, angle,
            velocity, ref_pos, n_chunk=1):
        input_data = dict(zip(TAGS, [img, reward, output,
                          position, angle, velocity, ref_pos]))
        with h5py.File(PATH, mode='a') as f:
            for tag in TAGS:
                print("tag", tag)
                print("type", type(input_data[tag]))
                try:
                    print("shape", input_data[tag].shape)
                except:
                    print("exception occurred")
                f[tag].resize(f[tag].shape[0]+n_chunk, axis=0)
                f[tag][-n_chunk:] = input_data[tag]
                f.flush()
