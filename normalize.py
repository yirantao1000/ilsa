import numpy as np
import argparse
import os
import random

# Define the path to the original file and number of actions to keep
parser = argparse.ArgumentParser()
parser.add_argument("--folder", default="data/pill/raw", type=str)
parser.add_argument("--target_folder", default="data/pill/normalized", type=str)
args = parser.parse_args()


limit = True

if not os.path.exists(args.target_folder):
    os.makedirs(args.target_folder)

def Normalize(data, min_data, max_data, min=-1, max=1):
    assert (min_data < max_data).all()
    data_std = (data - min_data) / (max_data - min_data)
    data_scaled = data_std * (max - min) + min
    return data_scaled

proprios_dim = 5
actions_dim = 5



min_proprios = np.array([np.inf]*proprios_dim)
max_proprios = np.array([-np.inf]*proprios_dim)

min_actions = np.array([np.inf]*actions_dim)
max_actions = np.array([-np.inf]*actions_dim)



file_count = 0
#compute range of proprios/actions
for filename in os.listdir(args.folder):
    if '.npz' in filename:
        file_count += 1
        # print(filename)
        full_path = os.path.join(args.folder, filename)
        normalized_path = os.path.join(args.target_folder, filename)

        proprios = np.load(full_path)["proprio"]
        actions = np.load(full_path)["action"]
        assert (-180 < proprios).all() and (proprios < 180).all()

        min_proprios = np.minimum(proprios[:,:-1].min(axis=0), min_proprios)   
        max_proprios = np.maximum(proprios[:,:-1].max(axis=0), max_proprios)   

        min_actions = np.minimum(actions.min(axis=0), min_actions)   
        
        max_actions = np.maximum(actions.max(axis=0), max_actions)   
print(min_proprios)
print(max_proprios)
print(min_actions)
print(max_actions)



if 'pill' in args.folder:
    min_objects = np.array([min_proprios[0], min_proprios[1], min_proprios[2], min_proprios[3], min_proprios[4], min_proprios[0], min_proprios[1], min_proprios[2], min_proprios[3], min_proprios[4]])
    max_objects = np.array([max_proprios[0], max_proprios[1], max_proprios[2], max_proprios[3], max_proprios[4], max_proprios[0], max_proprios[1], max_proprios[2], max_proprios[3], max_proprios[4]])
elif 'cereal' in args.folder:
    min_objects = np.array([min_proprios[0], min_proprios[1], min_proprios[2], min_proprios[4], min_proprios[0], min_proprios[1], min_proprios[2], min_proprios[4]])
    max_objects = np.array([max_proprios[0], max_proprios[1], max_proprios[2], max_proprios[4], max_proprios[0], max_proprios[1], max_proprios[2], max_proprios[4]])
else:
    raise NotImplementedError

for filename in os.listdir(args.folder):
    if '.npz' in filename:
        demo_index = int(filename[-5])

        full_path = os.path.join(args.folder, filename)
        normalized_path = os.path.join(args.target_folder, filename)

        proprios = np.load(full_path)["proprio"]
        actions = np.load(full_path)["action"]
        object = np.load(full_path)["object"] 

        assert (-180 < proprios).all() and (proprios < 180).all()

      
        
        proprios[:,:-1] = Normalize(proprios[:,:-1], min_proprios, max_proprios)
        actions = Normalize(actions, min_actions, max_actions)
        object = Normalize(object, min_objects, max_objects)      
        object = np.tile(object, (proprios.shape[0], 1))

  
        assert np.all((proprios >= -1) & (proprios <= 1))
        assert np.all((actions >= -1) & (actions <= 1))
    
       
       
        np.savez_compressed(normalized_path, proprio = proprios, action = actions, object = object)





    
           

            

            
           




