import sys
import os
import random
import numpy as np
import threading
import math

from kinova_basics import *

init_pos = []
limit = True

x_range = [] #robot workspace
y_range = []

handle_height_ranges = [
    [],
    [],
    [],
]
bottle_height_range = []
drawer_orient_range = []
out_range = []
out_ori_handle_range = []

generate_num = 50




def main():
           
    i=0
    while i < generate_num:
        print(i)

        handle_index = random.sample([0,1,2], 1)[0]
        handle_pos = [random.uniform(x_range[0], x_range[1]),
                    random.uniform(y_range[0], y_range[1]),
                    random.uniform(handle_height_ranges[handle_index][0], handle_height_ranges[handle_index][1]),
                    120,
                    random.uniform(drawer_orient_range[0], drawer_orient_range[1])]

        ball_pos = [random.uniform(x_range[0], x_range[1]),
                    random.uniform(y_range[0], y_range[1]),
                    random.uniform(bottle_height_range[0], bottle_height_range[1]),
                    165,
                    handle_pos[3]]


        pulled_value = random.uniform(out_range[0], out_range[1])
        pulled_x = handle_pos[0] - math.sin(math.radians(handle_pos[3])) * pulled_value
        pulled_y = handle_pos[1] + math.cos(math.radians(handle_pos[3])) * pulled_value
        pulled_pos = [pulled_x,
                    pulled_y,
                    handle_pos[2],
                    120,
                    handle_pos[3]]

        out_value = random.uniform(out_ori_handle_range[0], out_ori_handle_range[1])
        final_x = handle_pos[0] - math.sin(math.radians(handle_pos[3])) * out_value
        final_y = handle_pos[1] + math.cos(math.radians(handle_pos[3])) * out_value
        final_z = handle_pos[2] + 0.045
        
        final_pos = [final_x,
                    final_y,
                    final_z,
                    165,
                    handle_pos[3]]
            
        for process in ["open", "pick", "throw"]:
            previous_proprio = None
            file_path = f'data/pill/raw/{process}{i}.npz'
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            all_proprio = []
            all_action = []
            print(process)
            if process == "open":
                gripper = 0
                start_pos = init_pos
                end_pos = handle_pos
                
            elif process == "pick":
                gripper = 0
                start_pos = pulled_pos
                end_pos = ball_pos

            else:
                assert process == "throw"
                gripper = 1                   
                start_pos = ball_pos
                end_pos = final_pos

            start_pos = np.array(start_pos)
            end_pos = np.array(end_pos)

            distance = np.linalg.norm(start_pos[:3] - end_pos[:3])

            interpolation = int(np.ceil(distance/0.005))
            
            linspace_each_dimension = [np.linspace(start, end, interpolation) for start, end in zip(start_pos, end_pos)]
            interpolated_pos = np.stack(linspace_each_dimension)
            interpolated_pos = interpolated_pos.T
                            
            for j in range(interpolation):
                pos = interpolated_pos[j]
                                
                proprio = np.append(pos,gripper)
                                    
                if previous_proprio is None:
                    previous_proprio = proprio
                else:
                    action = proprio[:5] - previous_proprio[:5]
                    # if process == 'throw':
                    #     assert action[2]>=0
                    # min_actions = np.minimum(action, min_actions)   
                    # max_actions = np.maximum(action, max_actions) 
                    all_proprio.append(previous_proprio)
                    all_action.append(action)
                    previous_proprio = proprio

            all_proprio.append(proprio)
            all_action.append(np.array([0,0,0,0,0]))

            all_proprio = np.array(all_proprio)
            all_action = np.array(all_action)
            
            # if process == 'open':
            #     object = np.array([handle_pos[0], handle_pos[1], handle_pos[2], handle_pos[3]])
            # elif process == "pick":
            #     object = np.array([ball_pos[0], ball_pos[1], ball_pos[2], ball_pos[3]])
            # else:
            #     assert process == 'throw'
            #     object = np.array([final_pos[0], final_pos[1], final_pos[2], final_pos[3]])
            object = np.array([start_pos[0], start_pos[1], start_pos[2], start_pos[3], start_pos[4], end_pos[0], end_pos[1], end_pos[2], end_pos[3], end_pos[4]])
            np.savez(file_path, proprio=all_proprio, action=all_action, object=object)
        i+=1
            
    exit()
            
      

if __name__ == "__main__":
    exit(main())