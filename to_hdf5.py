import numpy as np
import h5py
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_name", default='cereal', type=str, help="")
args = parser.parse_args()

data_name = args.data_name 
hdf5_file_path = f'./data/{data_name}/kinematics.hdf5'
if "cereal" in hdf5_file_path:
    processes = ["pick", "pour"]
elif "pill" in hdf5_file_path:
    processes = ["open", "pick", "throw"]
else:
    raise NotImplementedError

finish = True

demo_num = 50
   
with h5py.File(hdf5_file_path, 'w') as f:
    data_group = f.create_group('data')
    mask_group = f.create_group('mask')
 
    indexes = list(range(demo_num))
    
    train_names = np.array([f"demo_{i}" for i in range(demo_num)], dtype='S')
    valid_names = np.array([], dtype='S')

    mask_group.create_dataset('train', data=train_names)
    mask_group.create_dataset('valid', data=valid_names)
    
    for i in indexes:
        print(i)
        proprios_whole = []
        objects_whole = []
        actions_whole = []

        for p in processes:
            file_path = f"./data/{data_name}/normalized/{p}{i}.npz".format(i)
            assert os.path.exists(file_path)
            proprios = np.load(file_path)["proprio"]
            objects = np.load(file_path)["object"]
            actions = np.load(file_path)["action"]

            if finish:
                finish_indicator = np.zeros((actions.shape[0], 1))
                finish_indicator[-1] = 1
                actions = np.hstack((actions, finish_indicator))
            proprios_whole.append(proprios)
            objects_whole.append(objects)
            actions_whole.append(actions)

        proprios_whole = np.vstack((proprios_whole))
        objects_whole = np.vstack((objects_whole))
        actions_whole = np.vstack((actions_whole))
        assert proprios_whole.shape[0] == actions_whole.shape[0] == objects_whole.shape[0]
 
        demo_group = data_group.create_group('demo_{}'.format(i))

        demo_group.attrs["num_samples"] = actions_whole.shape[0]

        obs_group = demo_group.create_group('obs')

        
        obs_group.create_dataset('proprios', data=proprios_whole)
        obs_group.create_dataset('objects', data=objects_whole)         
        demo_group.create_dataset('actions', data=actions_whole)

