import random
import numpy as np
import torch.nn.functional as F
import h5py
import os

from kinova_basics import *

adjust_slow = True
slow_speed = 0.004

task = 
epoch_index = 
interact_indices = 
interact_indices_to_merge = 

use_ori = False
from_init = 50
save_index = None 

fixed_inits = None

sample_point = None
sample_furthest = True
sample_closest = False
if sample_point is not None:
    assert sample_furthest ^ sample_closest


loc_only = True
latest = True
upgraded = False

if task == 'pill':
    processes = ['open', 'pick', 'throw']
    ac_dim = 5
elif task == 'cereal':
    processes = ['pick', 'pour']
    ac_dim = 5
else:
    raise NotImplementedError

task_dir = task


extract_action_from_delta_pro = True




def find_min_max():
    global min_objects
    global max_objects
    if task == 'pill':
        min_proprios = np.array([])

        max_proprios = np.array([])

        min_actions = np.array([])

        max_actions = np.array([])
            
        min_objects = np.array([min_proprios[0], min_proprios[1], min_proprios[2], min_proprios[3], min_proprios[4], min_proprios[0], min_proprios[1], min_proprios[2], min_proprios[3], min_proprios[4]])
        max_objects = np.array([max_proprios[0], max_proprios[1], max_proprios[2], max_proprios[3], max_proprios[4], max_proprios[0], max_proprios[1], max_proprios[2], max_proprios[3], max_proprios[4]])
    elif task == 'cereal':        
        min_proprios = np.array([])

        max_proprios = np.array([])

        min_actions = np.array([])

        max_actions = np.array([])

        
        min_objects = np.array([min_proprios[0], min_proprios[1], min_proprios[2], min_proprios[4], min_proprios[0], min_proprios[1], min_proprios[2], min_proprios[4]])
        max_objects = np.array([max_proprios[0], max_proprios[1], max_proprios[2], max_proprios[4], max_proprios[0], max_proprios[1], max_proprios[2], max_proprios[4]])

        
    else:
        raise NotImplementedError

    return min_proprios, max_proprios, min_actions, max_actions

min_proprios, max_proprios, min_actions, max_actions = find_min_max()


def Normalize(data, min_data, max_data, min=-1, max=1):
    assert (min_data < max_data).all()
    data_std = (data - min_data) / (max_data - min_data)
    data_scaled = data_std * (max - min) + min
    return data_scaled

def Unnormalize(data, min_data, max_data, min=-1, max=1):
    data_std = (data - min) / (max - min)
    original_data = data_std * (max_data - min_data) + min_data 
    return original_data



def adjust_multiple_source(all_actions, correct_mask, delta_proprios, ratio):
    n = delta_proprios.shape[0]
    i = 0
    sheer_delta_sequence = []
    modified_action_sequence_unnormalized = []
    first_correct = False
    while i < n:
    # Start of a new section
        if correct_mask[i] == 1:
            if not first_correct:
                first_correct = True
                if i != 0:
                    print("first action is wrong")
                    print("first wrong num: ", i)
                    global first_wrong
                    global first_wrong_num
                    first_wrong = True
                    first_wrong_num = i
                    sheer_delta_sequence.append(delta_proprios[:i])
                    
            start_correct = i
            # Find the end of this section
            while i < n and correct_mask[i] == 1:
                i += 1
            if i==n:
                print("last segment of actions are correct")

                sheer_delta_sequence.append(delta_proprios[start_correct:])
                modified_action_sequence_unnormalized.append(Unnormalize(all_actions[start_correct:], min_actions, max_actions))
                # for i in range(len(sheer_delta_sequence)):
                #     print(sheer_delta_sequence[i].shape)
                #     print(modified_action_sequence_unnormalized[i].shape)
                # exit()
                sheer_delta_sequence = np.vstack(sheer_delta_sequence)   
                
                modified_action_sequence_unnormalized = np.vstack(modified_action_sequence_unnormalized)
                # assert sheer_delta_sequence.shape[0] == modified_action_sequence_unnormalized.shape[0],(sheer_delta_sequence.shape[0],modified_action_sequence_unnormalized.shape[0])
                
                print(f"sheer data length:{sheer_delta_sequence.shape[0]}")
                return delta_proprios, sheer_delta_sequence, modified_action_sequence_unnormalized
            source_start = end_correct = i #exclusive
            # modified_first = 0
         
            while i < n and correct_mask[i] == 0:
                i += 1
            # while i<n:
            #     i+=1
                
            sum_delta_proprios = np.sum(delta_proprios[source_start:i], axis=0) #(4,)
            # print(sum_delta_proprios)
            
            # sum_d_proprios = sum_d_proprios[:, np.newaxis] #(4,1)
            move_ratio = ratio[start_correct:end_correct, 0] / np.sum(ratio[start_correct:end_correct, 0]) #(target_count,)
            allocated_delta_proprios = np.outer(move_ratio, sum_delta_proprios) #(target_count,4)
            assert np.allclose(np.sum(allocated_delta_proprios, axis=0), sum_delta_proprios)

            source_num = i - source_start
            target_num = end_correct - start_correct
            # print(f"i:{i}")
            # print(f"source_num:{source_num}")
            # print(f"target_num:{target_num}")
            # print("")

            delta_proprios[start_correct:end_correct] += allocated_delta_proprios

            
            sheer_delta_sequence.append(delta_proprios[start_correct:end_correct])
            modified_action_sequence_unnormalized.append(delta_proprios[start_correct:end_correct] / ratio[start_correct:end_correct])
                        
        else:
            assert not first_correct
            i += 1
    sheer_delta_sequence = np.vstack(sheer_delta_sequence)
    modified_action_sequence_unnormalized = np.vstack(modified_action_sequence_unnormalized)
    assert sheer_delta_sequence.shape[0] == modified_action_sequence_unnormalized.shape[0] or sheer_delta_sequence.shape[0] == modified_action_sequence_unnormalized.shape[0] + first_wrong_num
    print(f"sheer data length:{sheer_delta_sequence.shape[0]}")
    return delta_proprios, sheer_delta_sequence, modified_action_sequence_unnormalized



def merge_actions(actions):
    raise NotImplementedError
    magnitudes = np.linalg.norm(actions, axis=1)
    
    # Print all magnitudes
    print("Action Magnitudes:")
    for i, mag in enumerate(magnitudes):
        print(f"Action {i+1}/{actions.shape[0]}: {mag}")
    return
    # exit()

    # Initialize an empty list to store the combined actions
    combined_actions = []
    
    # Initialize a temporary storage for summing consecutive below-threshold actions
    temp_action = np.zeros(3)
    combining = False

    for i in range(len(magnitudes)):
        if magnitudes[i] < threshold:
            temp_action += actions[i]
            combining = True
        else:
            if combining:
                combined_actions.append(temp_action)
                temp_action = np.zeros(3)
                combining = False
            combined_actions.append(actions[i])
    
    if combining:
        combined_actions.append(temp_action)

    # Convert the combined actions back to a NumPy array
    combined_actions = np.array(combined_actions)
    
    return combined_actions



def generate_modified_proprios(init_proprios, delta_sequence,gripper):
    # print(init_proprios)
    # print(delta_sequence)
    # exit()
    
    #init_proprios and delta_sequence are both unnormalized
    modified_proprios = []
    current_pos = init_proprios[:ac_dim]
    
    pos = [current_pos[0],current_pos[1],current_pos[2],current_pos[3],0,current_pos[4]]
    
    
    current_proprios = np.concatenate((current_pos,np.array([gripper])))
    
    
    if loc_only:
        assert current_proprios.shape[-1] == ac_dim + 1, current_proprios.shape[-1]
    else:
        assert current_proprios.shape[-1] == ac_dim + 7, current_proprios.shape[-1]
    while np.any(current_proprios[3:-1] < -180) or np.any(current_proprios[3:-1] > 180):
        current_proprios[3:-1][current_proprios[3:-1] > 180] -= 360
        current_proprios[3:-1][current_proprios[3:-1] < -180] += 360
    # current_proprios[3:-1][current_proprios[3:-1] > 180] -= 360
    # current_proprios[3:-1][current_proprios[3:-1] < -180] += 360
    current_proprios[:-1] = Normalize(current_proprios[:-1], min_proprios, max_proprios)
    modified_proprios.append(current_proprios)
    for i in range(delta_sequence.shape[0]):
        # print(f'{i}/{delta_sequence.shape[0]}')
        current_pos += delta_sequence[i]
        # print(delta_sequence[i])
        # print(i)
        # print(current_pos)
        pos = [current_pos[0],current_pos[1],current_pos[2],current_pos[3],0,current_pos[4]]

        current_proprios = np.concatenate((current_pos,np.array([gripper])))
        
        # print(current_proprios)
        while np.any(current_proprios[3:-1] < -180) or np.any(current_proprios[3:-1] > 180):
            current_proprios[3:-1][current_proprios[3:-1] > 180] -= 360
            current_proprios[3:-1][current_proprios[3:-1] < -180] += 360
        current_proprios[:-1] = Normalize(current_proprios[:-1], min_proprios, max_proprios)
        modified_proprios.append(current_proprios)
    return np.vstack(modified_proprios)

def process_single_demo(all_paths, interact_index):
    proprios_whole_process_ori = []
    objects_whole_process_ori = []
    actions_whole_process_ori = []
    user_actions_whole_process_ori = []
    corrects_whole_process_ori = []

    proprios_whole_process_modified = []
    objects_whole_process_modified = []
    actions_whole_process_modified = []
    user_actions_whole_process_modified = []
    corrects_whole_process_modified = []

    for process in processes:
        global first_wrong
        first_wrong = False
        name = f'{interact_index}_{process}'
        print(name)
        path = all_paths[processes.index(process)]
        if 'pick' in path or 'open' in path:
            gripper = 0
        else:
            assert 'pour' in path or 'throw' in path
            gripper = 1
        all_file = h5py.File(path, 'r') 

        demo_keys = list(all_file['data'].keys())
        # print(list(demo_keys)[0])
        assert len(demo_keys)==1, demo_keys
        all_data = all_file['data'][demo_keys[0]]

        all_proprios = np.array(all_data['obs']['proprios'])       
        all_objects = np.array(all_data['obs']['objects'])
        all_actions = np.array(all_data['actions'])
        if 'pour' in process:
            assert len(all_actions.shape)==2 and all_actions.shape[-1]==6
            all_actions[:,3:5] = Normalize(np.array([0,0]),min_actions[3:5],max_actions[3:5])
        all_user_actions = np.array(all_data['user_actions'])
        correct_mask = np.array(all_data['corrects'])

        lines_to_delete = []
        #delete repetitiev data
        for i in range(1, all_proprios.shape[0]):
            # print(f'{i}/{all_proprios.shape[0]}')
            if (all_proprios[i,:ac_dim]==all_proprios[i-1,:ac_dim]).all():
                lines_to_delete.append(i)
        if len(lines_to_delete)>0:
            for i in lines_to_delete:
                all_proprios = np.delete(all_proprios, i, axis=0)
                all_objects = np.delete(all_objects, i, axis=0)
                all_actions = np.delete(all_actions, i, axis=0)
                all_user_actions = np.delete(all_user_actions, i, axis=0)
                correct_mask = np.delete(correct_mask, i, axis=0)


        all_num = all_proprios.shape[0]
        assert all_num == all_objects.shape[0] == all_actions.shape[0]+1
        assert all_actions.shape[1]==ac_dim+1
        all_actions_finish_included = all_actions.copy()
        all_actions = all_actions[:,:-1]
        

        rotate_indices = np.array([])
        non_rotate_indices = np.arange(all_user_actions.shape[0])
        assert rotate_indices.shape[0] == 0
        assert rotate_indices.shape[0]+non_rotate_indices.shape[0] == all_user_actions.shape[0]
        # print("rotate user actions", all_user_actions[non_rotate_indices])
        all_user_actions = all_user_actions[:,:-1]
        

        number_of_ones = np.count_nonzero(correct_mask == 1)
        correct_indices = np.where(correct_mask == 1)[0]
        


        print(f"Correct data: {number_of_ones}")
        print(f"Wrong data: {all_num - number_of_ones}")
        print(f"User rotate data: {rotate_indices.shape[0]}")

        ratio = []
        delta_proprios = []

        #calculate delta proprios and ratios
        for i in range(all_num-1):
            p0 = Unnormalize(all_proprios[i,:ac_dim], min_proprios[:ac_dim], max_proprios[:ac_dim])
            p1 = Unnormalize(all_proprios[i+1,:ac_dim], min_proprios[:ac_dim], max_proprios[:ac_dim])
            if correct_mask[i]:           
                # print(p1-p0)
                a = Unnormalize(all_actions[i,:ac_dim], min_actions[:ac_dim], max_actions[:ac_dim])
                # print(a) 
                # if (((p1-p0)/a) == 0).all():
                #     continue
                # assert not (((p1-p0)/a) == 0).all(), (i, all_proprios[i], all_proprios[i+1])
                ratio.append((p1-p0)/a)          
            else:
                # assert all_user_actions[i]<=5
                # a = mapping[all_user_actions[i]]
                ratio.append(np.array([0]*ac_dim))
            # ratio.append((p1-p0)/a)
            # ratio.append((p1[0]-p0[0])/a)
            delta_proprios.append(p1-p0)
        
        # if process == 'pour':
        #     print(delta_proprios)
        #     exit()

        delta_proprios = np.array(delta_proprios)
        ratio = np.array(ratio)
        assert len(delta_proprios.shape)==2
        assert len(ratio.shape)==2
        
        delta_proprios, sheer_delta_sequence, modified_action_sequence_unnormalized = adjust_multiple_source(all_actions, correct_mask, delta_proprios, ratio)
        
        

        # print(all_path)
        
        proprios_to_save = np.vstack([Unnormalize(all_proprios[0,:ac_dim], min_proprios[:ac_dim], max_proprios[:ac_dim]), sheer_delta_sequence])
        proprios_save_path = f'./data/{task_dir}/incremental/{epoch_index}/modified/unnormalized_proprios/{name}.npy'
        if not os.path.exists(os.path.dirname(proprios_save_path)):
            os.makedirs(os.path.dirname(proprios_save_path))

        np.save(proprios_save_path, proprios_to_save)

        if extract_action_from_delta_pro:
            unnormalized_actions = Unnormalize(all_actions, min_actions, max_actions)
            ori_correct_mag = np.linalg.norm(unnormalized_actions[correct_mask.astype(bool)][:, :3], axis=1)
            if first_wrong:
                print("first wrong num: ", first_wrong_num)
                ori_correct_mag = np.concatenate((np.full(first_wrong_num, 0.005), ori_correct_mag))
            magnitudes = np.linalg.norm(sheer_delta_sequence[:, :3], axis=1)
            assert ori_correct_mag.shape[-1] == magnitudes.shape[-1], (first_wrong, first_wrong_num, ori_correct_mag.shape[-1], magnitudes.shape[-1])
            if adjust_slow:
                print(f"Adjusting incorrect slow actions: {np.sum(ori_correct_mag < slow_speed)}/{ori_correct_mag.shape[0]}")
                ori_correct_mag[ori_correct_mag < slow_speed] = slow_speed

            assert ori_correct_mag.shape[0]==magnitudes.shape[0], (ori_correct_mag.shape, magnitudes.shape)
            scaling_factors = ori_correct_mag / magnitudes
            modified_action_sequence_unnormalized = sheer_delta_sequence * scaling_factors[:, np.newaxis]
            
        else:
            raise NotImplementedError
            merge_actions(modified_action_sequence_unnormalized)



        modified_actions_unnormalized = np.vstack((modified_action_sequence_unnormalized, np.zeros((1, modified_action_sequence_unnormalized.shape[1]))))
        
        modified_actions= Normalize(modified_actions_unnormalized,min_actions,max_actions)
        finish_indicators = np.zeros((modified_actions.shape[0], 1))
        finish_indicators[-1] = 1
        modified_actions = np.hstack((modified_actions, finish_indicators))

        
        proprios_whole_process_ori.append(all_proprios[:-1][non_rotate_indices])
        objects_whole_process_ori.append(all_objects[:-1][non_rotate_indices])
        actions_whole_process_ori.append(all_actions_finish_included[non_rotate_indices])
        user_actions_whole_process_ori.append(all_user_actions[non_rotate_indices])
        corrects_whole_process_ori.append(correct_mask[non_rotate_indices])
        assert proprios_whole_process_ori[-1].shape[0] == objects_whole_process_ori[-1].shape[0] == actions_whole_process_ori[-1].shape[0] == user_actions_whole_process_ori[-1].shape[0] == corrects_whole_process_ori[-1].shape[0]
        
        modified_proprios = generate_modified_proprios(Unnormalize(all_proprios[0,:ac_dim], min_proprios[:ac_dim], max_proprios[:ac_dim]),sheer_delta_sequence,gripper)
        # assert modified_actions.shape[0]==modified_proprios.shape[0]==proprios_whole_process_ori[-1].shape[0]+1
        modified_objects = np.tile(all_objects[0], (modified_proprios.shape[0], 1))
        if first_wrong:
            modified_user_actions = np.vstack((all_user_actions[:first_wrong_num], all_user_actions[correct_indices]))
        else:
            modified_user_actions = all_user_actions[correct_indices]
        modified_user_actions = np.vstack((modified_user_actions, modified_user_actions[-1][np.newaxis, :]))
        assert modified_user_actions.shape[0]==modified_objects.shape[0]==modified_proprios.shape[0]
        
        
        modified_corrects = np.ones(modified_proprios.shape[0])
        proprios_whole_process_modified.append(modified_proprios)
        objects_whole_process_modified.append(modified_objects)
        actions_whole_process_modified.append(modified_actions)
        user_actions_whole_process_modified.append(modified_user_actions)
        corrects_whole_process_modified.append(modified_corrects)
        assert proprios_whole_process_modified[-1].shape[0] == objects_whole_process_modified[-1].shape[0] == actions_whole_process_modified[-1].shape[0] == user_actions_whole_process_modified[-1].shape[0] == corrects_whole_process_modified[-1].shape[0]

 
    proprios_whole_ori = np.concatenate([proprio for proprio in proprios_whole_process_ori], axis=0)
    objects_whole_ori = np.concatenate([obj for obj in objects_whole_process_ori], axis=0)
    actions_whole_ori = np.concatenate([action for action in actions_whole_process_ori], axis=0)
    user_actions_whole_ori = np.concatenate([user_action for user_action in user_actions_whole_process_ori], axis=0)
    corrects_whole_ori = np.concatenate([correct for correct in corrects_whole_process_ori], axis=0)
    modified_whole_ori = np.concatenate([np.zeros(proprio.shape[0]) for proprio in proprios_whole_process_ori], axis=0)

    
    proprios_whole_modified = np.concatenate([proprio for proprio in proprios_whole_process_modified], axis=0)
    objects_whole_modified = np.concatenate([obj for obj in objects_whole_process_modified], axis=0)
    actions_whole_modified = np.concatenate([action for action in actions_whole_process_modified], axis=0)
    user_actions_whole_modified = np.concatenate([user_action for user_action in user_actions_whole_process_modified], axis=0)
    corrects_whole_modified = np.concatenate([correct for correct in corrects_whole_process_modified], axis=0)
    modified_whole_modified = np.concatenate([np.ones(proprio.shape[0]) for proprio in proprios_whole_process_modified], axis=0)

 
    print(f"ori data length: {proprios_whole_ori.shape[0]}")
    print(f"modified data length: {proprios_whole_modified.shape[0]}")

    return proprios_whole_ori, objects_whole_ori, actions_whole_ori, user_actions_whole_ori, corrects_whole_ori, modified_whole_ori,\
            proprios_whole_modified, objects_whole_modified, actions_whole_modified, user_actions_whole_modified, corrects_whole_modified, modified_whole_modified
    

def save_modified(index, save_path, proprios_whole, objects_whole, actions_whole, user_actions_whole, corrects_whole, modified_whole):
    with h5py.File(save_path, 'w') as f:
        # Create the necessary groups and datasets
        data_group = f.create_group('data')

        mask_group = f.create_group('mask')
        train_names = np.array(["demo_{}".format(index)], dtype='S')
        valid_names = np.array([], dtype='S')

        mask_group.create_dataset('train', data=train_names)
        mask_group.create_dataset('valid', data=valid_names)

        demo_group = data_group.create_group('demo_{}'.format(index))
        obs_group = demo_group.create_group('obs')

    
        demo_group.attrs["num_samples"] = actions_whole.shape[0]

        obs_group.create_dataset('proprios', data=proprios_whole)
        obs_group.create_dataset('objects', data=objects_whole)
        demo_group.create_dataset('actions', data=actions_whole)
        demo_group.create_dataset('user_actions', data=user_actions_whole)

        demo_group.create_dataset('corrects', data=corrects_whole)
        demo_group.create_dataset('modified', data=modified_whole)
        
        print("{} saved".format(save_path))


first_wrong = False


# save new demos (both ori and modified)
for index in interact_indices:
    modified_path = f'./data/{task_dir}/incremental/{epoch_index}/modified_{index}.hdf5'
    with h5py.File(modified_path, 'w') as f:
        data_group = f.create_group('data')

        mask_group = f.create_group('mask')
        # train_names = np.array(["demo_{}".format(interact_index) for interact_index in range(interact_indices[0],2*(interact_indices[-1]+1))], dtype='S')
        train_names = np.array(["demo_{}".format(interact_index) for interact_index in [2*index, 2*index+1]], dtype='S')
        print(train_names)
        valid_names = np.array([], dtype='S')

        mask_group.create_dataset('train', data=train_names)
        mask_group.create_dataset('valid', data=valid_names)


    
        demo_group_ori = data_group.create_group('demo_{}'.format(index*2))
        demo_group_modified = data_group.create_group('demo_{}'.format(index*2+1))
        obs_group_ori = demo_group_ori.create_group('obs')
        obs_group_modified = demo_group_modified.create_group('obs')

        all_paths = []

        for process in processes:
            all_paths.append(f'./data/{task_dir}/incremental/{epoch_index}/ori/{index}_{process}.hdf5')


         
        proprios_whole_ori, objects_whole_ori, actions_whole_ori, user_actions_whole_ori, corrects_whole_ori, modified_whole_ori,\
            proprios_whole_modified, objects_whole_modified, actions_whole_modified, user_actions_whole_modified, corrects_whole_modified, modified_whole_modified = process_single_demo(all_paths, index)
        # save_path = all_paths[0].replace(f'ori/all_pick{index}.hdf5',f'modified/{index}.hdf5')
        # save_modified(index, save_path, proprios_whole, objects_whole, actions_whole, user_actions_whole, corrects_whole, modified_whole)

        demo_group_ori.attrs["num_samples"] = proprios_whole_ori.shape[0]
        demo_group_modified.attrs["num_samples"] = proprios_whole_modified.shape[0]

                        
        obs_group_ori.create_dataset('proprios', data=proprios_whole_ori)
        obs_group_ori.create_dataset('objects', data=objects_whole_ori)

        obs_group_modified.create_dataset('proprios', data=proprios_whole_modified)
        obs_group_modified.create_dataset('objects', data=objects_whole_modified)

        demo_group_ori.create_dataset('actions', data=actions_whole_ori)
        demo_group_ori.create_dataset('user_actions', data=user_actions_whole_ori)

        demo_group_modified.create_dataset('actions', data=actions_whole_modified)
        demo_group_modified.create_dataset('user_actions', data=user_actions_whole_modified)

        demo_group_ori.create_dataset('corrects', data=corrects_whole_ori)
        demo_group_ori.create_dataset('modified', data=modified_whole_ori)

        demo_group_modified.create_dataset('corrects', data=corrects_whole_modified)
        demo_group_modified.create_dataset('modified', data=modified_whole_modified)

    print("{} saved".format(modified_path))


all_indices = interact_indices + interact_indices_to_merge
if len(interact_indices_to_merge) > 0:
    smallest_index = interact_indices_to_merge[0]
else:
    smallest_index = interact_indices[0]
if len(interact_indices)>0:
    largest_index = interact_indices[-1]
else:
    largest_index = interact_indices_to_merge[-1]

# if len(all_indices)==1:
# if from_init > 0:
#     assert smallest_index != 0
#     # assert smallest_index == largest_index ==1
#     smallest_index = 0

if use_ori: 
    save_name = f'modified_{smallest_index}-{largest_index}+{from_init}init_w_ori' 
else:
    save_name = f'modified_{smallest_index}-{largest_index}+{from_init}init'

if save_index is not None:
    save_name += f'_{save_index}'

if sample_point is not None:
    if sample_closest:
        save_name += '_closest'
    if sample_furthest:
        save_name += '_furthest'

if fixed_inits is not None:
    save_name += '_fixed_inits'



modified_all_path = f'./data/{task_dir}/incremental/{epoch_index}/{save_name}.hdf5'       


    
with h5py.File(modified_all_path, 'w') as f:
    data_group = f.create_group('data')
    mask_group = f.create_group('mask')
    # train_names = np.array(["demo_{}".format(interact_index) for interact_index in range(interact_indices[0],2*(interact_indices[-1]+1))], dtype='S')
    if use_ori:
        train_indices = list(range(2*smallest_index, 2*largest_index + 2))
    else:
        train_indices = list(range(2*smallest_index + 1, 2*largest_index + 2, 2))
    
    if from_init > 0:
        train_indices += list(range(2*largest_index + 2, 2*largest_index + 2 + from_init))

    train_names = np.array(["demo_{}".format(interact_index) for interact_index in train_indices], dtype='S')
    print(train_names)
    valid_names = np.array([], dtype='S')

    mask_group.create_dataset('train', data=train_names)
    mask_group.create_dataset('valid', data=valid_names)


    for index in all_indices:
        modified_path = f'./data/{task_dir}/incremental/{epoch_index}/modified_{index}.hdf5'
        demo_modified = h5py.File(modified_path, 'r')['data'][f'demo_{index*2+1}']

        # if use_wrong:
        #     use_mask = np.ones(demo_modified['corrects'].shape)
        # else:
        #     use_mask = demo_modified['corrects']
        #     print(f"Using {np.sum(use_mask)}/{use_mask.shape[0]} correct data")
            


        demo_group_modified = data_group.create_group('demo_{}'.format(index*2+1))
        obs_group_modified = demo_group_modified.create_group('obs')

        demo_group_modified.attrs["num_samples"] = demo_modified['obs']['proprios'].shape[0]

        obs_group_modified.create_dataset('proprios', data=demo_modified['obs']['proprios'])
        obs_group_modified.create_dataset('objects', data=demo_modified['obs']['objects'])

        demo_group_modified.create_dataset('actions', data=demo_modified['actions'])
        demo_group_modified.create_dataset('user_actions', data=demo_modified['user_actions'])
        np.set_printoptions(threshold=np.inf)
        # print(np.array(demo_modified['user_actions']))

        demo_group_modified.create_dataset('corrects', data=demo_modified['corrects'])
        demo_group_modified.create_dataset('modified', data=demo_modified['modified'])
        demo_group_modified.create_dataset('inits', data=np.zeros(demo_modified['modified'].shape))

        if use_ori:
            demo_ori = h5py.File(modified_path, 'r')['data'][f'demo_{index*2}']
            demo_group_ori = data_group.create_group('demo_{}'.format(index*2))

            # if use_wrong:
            #     use_mask = np.ones(demo_ori['corrects'].shape)               
            # else:
            #     use_mask = demo_ori['corrects']
            #     print(f"Using {np.sum(use_mask)}/{use_mask.shape[0]} correct data")


            obs_group_ori = demo_group_ori.create_group('obs')
            demo_group_ori.attrs["num_samples"] = demo_ori['obs']['proprios'].shape[0]

            obs_group_ori.create_dataset('proprios', data=demo_ori['obs']['proprios'])
            obs_group_ori.create_dataset('objects', data=demo_ori['obs']['objects'])

            demo_group_ori.create_dataset('actions', data=demo_ori['actions'])
            demo_group_ori.create_dataset('user_actions', data=demo_ori['user_actions'])

            demo_group_ori.create_dataset('corrects', data=demo_ori['corrects'])
            demo_group_ori.create_dataset('modified', data=demo_ori['modified'])
            demo_group_ori.create_dataset('inits', data=np.zeros(demo_ori['modified'].shape))


    if from_init > 0:
        init_path = f'./data/{task_dir}/corrected.hdf5'    
        assert os.path.exists(init_path)
        init_data = h5py.File(init_path, 'r')['data']
        sampled_objects = []
        sampled_indices = []

        if fixed_inits is not None:
            for d in range(50):
                for fixed_init in fixed_inits:
                    if np.allclose(init_data[f'demo_{d}']['obs']['objects'][-1], fixed_init, atol=1e-3):
                        sampled_indices.append(d)
            print(sampled_indices)
            for i in range(len(sampled_indices)):
                sampled_index = sampled_indices[i]
                demo = init_data[f'demo_{sampled_index}']
                demo_group = data_group.create_group(f'demo_{2*largest_index + 2 + i}')
                obs_group = demo_group.create_group('obs')
                demo_group.attrs["num_samples"] = demo['obs']['proprios'].shape[0]
                
                obs_group.create_dataset('proprios', data=demo['obs']['proprios'])
                obs_group.create_dataset('objects', data=demo['obs']['objects'])
                sampled_objects.append(demo['obs']['objects'][-1])
                
                demo_group.create_dataset('actions', data=demo['actions'])
                demo_group.create_dataset('user_actions', data=np.zeros((demo['obs']['proprios'].shape[0],3)))
                
                demo_group.create_dataset('corrects', data=np.ones(demo['obs']['proprios'].shape[0]))
                demo_group.create_dataset('modified', data=np.zeros(demo['obs']['proprios'].shape[0]))
                demo_group.create_dataset('inits', data=np.ones(demo['obs']['proprios'].shape[0]))


        elif sample_point is not None:
            init_objects = []
            for d in range(50):
                init_objects.append(init_data[f'demo_{d}']['obs']['objects'][-1])
            init_objects = np.array(init_objects)
            distances = np.linalg.norm(init_objects - sample_point, axis=1)
            
            if sample_closest:
                sampled_indices = np.argsort(distances)[:from_init]
            else:
                assert sample_furthest
                sampled_indices = np.argsort(distances)[-from_init:]
            print(sampled_indices)
            
            for i in range(sampled_indices.shape[0]):
                sampled_index = sampled_indices[i]
                demo = init_data[f'demo_{sampled_index}']
                demo_group = data_group.create_group(f'demo_{2*largest_index + 2 + i}')
                obs_group = demo_group.create_group('obs')
                demo_group.attrs["num_samples"] = demo['obs']['proprios'].shape[0]
                
                obs_group.create_dataset('proprios', data=demo['obs']['proprios'])
                obs_group.create_dataset('objects', data=demo['obs']['objects'])
                sampled_objects.append(demo['obs']['objects'][-1])
                
                demo_group.create_dataset('actions', data=demo['actions'])
                demo_group.create_dataset('user_actions', data=np.zeros((demo['obs']['proprios'].shape[0],3)))
                
                demo_group.create_dataset('corrects', data=np.ones(demo['obs']['proprios'].shape[0]))
                demo_group.create_dataset('modified', data=np.zeros(demo['obs']['proprios'].shape[0]))
                demo_group.create_dataset('inits', data=np.ones(demo['obs']['proprios'].shape[0]))


        else:

            for i in range(from_init):
                sampled_index = random.randint(0, 49)
                while sampled_index in sampled_indices:
                    sampled_index = random.randint(0, 49)
                sampled_indices.append(sampled_index)
                
                demo = init_data[f'demo_{sampled_index}']
                demo_group = data_group.create_group(f'demo_{2*largest_index + 2 + i}')
                obs_group = demo_group.create_group('obs')
                demo_group.attrs["num_samples"] = demo['obs']['proprios'].shape[0]
                
                obs_group.create_dataset('proprios', data=demo['obs']['proprios'])
                obs_group.create_dataset('objects', data=demo['obs']['objects'])
                sampled_objects.append(demo['obs']['objects'][-1])
                
                demo_group.create_dataset('actions', data=demo['actions'])
                demo_group.create_dataset('user_actions', data=np.zeros((demo['obs']['proprios'].shape[0],3)))
                
                demo_group.create_dataset('corrects', data=np.ones(demo['obs']['proprios'].shape[0]))
                demo_group.create_dataset('modified', data=np.zeros(demo['obs']['proprios'].shape[0]))
                demo_group.create_dataset('inits', data=np.ones(demo['obs']['proprios'].shape[0]))
        
        print('sampled objects: ', sampled_objects)

print("{} saved".format(modified_all_path))

    
