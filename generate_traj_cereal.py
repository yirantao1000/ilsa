import sys
import os
import random
import numpy as np
import threading
import math

from kinova_basics import *

init_pos = []
limit = True

x_range = []
y_range = []
shelf_orient_range = []
grasp_height_range = []

pour_shelf_height_range = []

cup_radius_range = []
bowl_radius_range = []

generate_num = 50


def main():
    i=0
    while i < generate_num:
        print(i)
        grasp_pos = [random.uniform(x_range[0], x_range[1]),
                    random.uniform(y_range[0], y_range[1]),
                    random.uniform(grasp_height_range[0], grasp_height_range[1]),
                    90,
                    random.uniform(shelf_orient_range[0], shelf_orient_range[1])]
        
        pour_height = random.uniform(pour_shelf_height_range[0], pour_shelf_height_range[1])
        orien = grasp_pos[4]
        
        pour_pos = [random.uniform(x_range[0], x_range[1]),
                    random.uniform(y_range[0], y_range[1]),
                    pour_height,
                    90,
                    orien]
            
        for process in ["pick", "pour"]:
            previous_proprio = None
            file_path = 'data/cereal/raw/{}{}.npz'.format(process, i)
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            all_proprio = []
            all_action = []
            print(process)
            if process == "pick":
                gripper = 0
                start_pos = init_pos
                end_pos = grasp_pos
                
            else:
                assert process == "pour"
                gripper = 1                   
                start_pos = grasp_pos
                end_pos = pour_pos

            start_pos = np.array(start_pos)
            end_pos = np.array(end_pos)

            distance = np.linalg.norm(start_pos[:3] - end_pos[:3])

            interpolation = int(np.ceil(distance/0.005))
            
            linspace_each_dimension = [np.linspace(start, end, interpolation) for start, end in zip(start_pos, end_pos)]
            interpolated_pos = np.stack(linspace_each_dimension)
            interpolated_pos = interpolated_pos.T
                            
            for j in range(interpolation):
                pos = interpolated_pos[j]
                # print(pos)

                proprio = np.append(pos,gripper)
                                    
                if previous_proprio is None:
                    previous_proprio = proprio
                else:
                    action = proprio[:5] - previous_proprio[:5]
                    
                    all_proprio.append(previous_proprio)
                    all_action.append(action)
                    previous_proprio = proprio

            all_proprio.append(proprio)
            all_action.append(np.array([0,0,0,0,0]))

            all_proprio = np.array(all_proprio)
            all_action = np.array(all_action)
            object = np.array([start_pos[0], start_pos[1], start_pos[2], start_pos[4], end_pos[0], end_pos[1], end_pos[2], end_pos[4]])
            
            np.savez(file_path, proprio=all_proprio, action=all_action, object=object)
        i+=1
            
    exit()
        
      

if __name__ == "__main__":
    exit(main())