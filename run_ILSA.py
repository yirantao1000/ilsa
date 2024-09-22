import numpy as np
import json
import math
import os
import h5py
from collections import deque
import datetime
import random
from collections import defaultdict
import robosuite.utils.robomimic.file_utils as FileUtils
import robosuite.utils.robomimic.torch_utils as TorchUtils
from kinova_basics import *
import pygame
pygame.init()
pygame.joystick.init()

# Check for joysticks
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
else:
    print("No joystick found.")
    exit()

import sys
import warnings
warnings.filterwarnings('ignore')
import cv2
import time
from get_localizations import *

class HistoryBuffer:
    def __init__(self, size=10):
        self.size = size
        self.buffer = deque(maxlen=size)

    def add(self, item):
        """Add an item to the FIFO buffer."""
        self.buffer.append(item)

    def get_all(self):
        """Return all items in the FIFO buffer."""
        return list(self.buffer)
    
    def is_empty(self):
        """Check if the FIFO buffer is empty."""
        return not self.buffer


npy_folder = '' 
npy_name = ''
ckpt_path = ''

interact_epoch = int(input("interact epoch: "))
interact_index = int(input("interact index: "))

preset_locations = False
config_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)),"config.json")

with open(config_path, 'r') as file:
    config = json.load(file)
    algo_config = config["algo"]
    xbox_threshold = algo_config["xbox_threshold"]

    try:
        latest = algo_config["latest"]
    except:
        latest = False

    try:
        upgraded = algo_config["upgraded"]
    except:
        upgraded = False

    try:
        loc_only = algo_config["loc_only"]
    except:
        loc_only = False
    
    try:
        finish = algo_config["finish"]
        finish_separate = algo_config["finish_separate"]
    except:
        finish = finish_separate = False


    transformer_enabled = algo_config["transformer"]["enabled"]
    assert transformer_enabled
    
    if transformer_enabled:
        frame_stack = config["train"]["frame_stack"]
        proprios_buffer = HistoryBuffer(size = frame_stack)
        objects_buffer = HistoryBuffer(size = frame_stack)


if task == 'cereal':
    processes = ['pick', 'pour'] #has to be in order
    shelf_orientation = None
    bowl_y = None
    shelf_height = 
    fixed_radius = True
    if fixed_radius:
        cup_radius = 
        bowl_radius = 
    shelf_orien_range = []
    close = 0.3
    open = 0

elif task == 'pill':
    drawer_orientation = None
    processes = ['open', 'pick', 'throw'] #has to be in order
    button_heights = []
    close = 1
    open = 0.5

else:
    raise NotImplementedError





scale = False
finish_by_distance = False
stop_distance = 0.01
stop_distance = 0.04
speed = 10 #only need when scale = False


finish_lower_bound = 0.8
finish_upper_bound = 1.0
finish_adjust_speed = 0.5
slow_speed = 0.0025 * speed
slow_time = 1
fast_speed = 0.005 * speed
fast_time = 0.5



pour_speed = 20.0
open_drawer_speed = 0.005 * speed
move_speed = 0.004 * speed #user move speed
fight_seconds_before_obey = 0
fight_seconds_after_obey = 0.4

fight_seconds = fight_seconds_before_obey
fight_stop_seconds = 0
obey_seconds = 0.3


move_time = 1/speed

agreement_threshold = 0.5
if agreement_threshold == 1:
    fight_seconds = 0

consecutive = 1


if preset_locations:
    object_ori_dict = get_objects_preset(task)
    print("object_ori_dict: ", object_ori_dict)
    input("correct?")
else:
    object_ori_dict = get_objects(task)
    print("object_ori_dict: ", object_ori_dict)
    input("correct?")
    




def find_min_max(task, loc_only):
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

min_proprios, max_proprios, min_actions, max_actions = find_min_max(task, loc_only)

def get_proprios(task, loc_only, gripper):
    feedback = base_cyclic.RefreshFeedback()
    if task == 'pill':
        proprios = np.array([feedback.base.tool_pose_x,
                feedback.base.tool_pose_y,
                feedback.base.tool_pose_z,
                feedback.base.tool_pose_theta_x,
                feedback.base.tool_pose_theta_z,
                gripper])

    elif task == 'cereal':       
        proprios = np.array([feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z,
            feedback.base.tool_pose_theta_x,
            feedback.base.tool_pose_theta_z,
            gripper])
        
    
    else:
        raise NotImplementedError

    return proprios



def Normalize(data, min_data, max_data, min=-1, max=1):
    assert (min_data < max_data).all()
    data_std = (data - min_data) / (max_data - min_data)
    data_scaled = data_std * (max - min) + min
    return data_scaled

def Unnormalize(data, min_data, max_data, min=-1, max=1):
    data_std = (data - min) / (max - min)
    original_data = data_std * (max_data - min_data) + min_data 
    return original_data

def cosine_similarity(vector1, vector2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    return dot_product / (magnitude1 * magnitude2)


def save_file():
    global current_process
    global all_proprios
    global all_objects
    global all_actions
    global all_user_actions
    global all_corrects

    global correct
    global wrong
    
    #print statistics
    correct_num_dict[current_process] = correct
    wrong_num_dict[current_process] = wrong
    correct = 0
    wrong = 0

    for p in range(processes.index(current_process)+1):
        print(processes[p])
        print(f"Correct actions: {correct_num_dict[processes[p]]}")
        print(f"Wrong actions: {wrong_num_dict[processes[p]]}")
        print(f"All actions: {correct_num_dict[processes[p]]+wrong_num_dict[processes[p]]}")
    print("object dict: ", object_ori_dict)
    if task == 'pill':
        print("drawer orientation: ", drawer_orientation)
    if task == 'cereal':
        print("shelf orientation: ", shelf_orientation)

    
    proprios = get_proprios(task, loc_only, gripper)
    

    proprios[3:-1][proprios[3:-1] > 180] -= 360
    proprios[3:-1][proprios[3:-1] < -180] += 360

    proprios[:-1] = Normalize(proprios[:-1], min_proprios[:proprios.shape[-1]-1], max_proprios[:proprios.shape[-1]-1])

    object_ori = object_ori_dict[current_process]
    
    objects = Normalize(object_ori, min_proprios[:object_ori.shape[0]], max_proprios[:object_ori.shape[0]])
    
    all_proprios.append(proprios)         
    all_objects.append(objects)     

    all_proprios = np.array(all_proprios)
    all_objects = np.array(all_objects)
    all_actions = np.array(all_actions)
    all_user_actions = np.array(all_user_actions)

   
    hdf5_file_path = f'./data/{task}/incremental/{interact_epoch}/ori/{interact_index}_{current_process}.hdf5'

    
    assert not os.path.exists(hdf5_file_path)
    if not os.path.exists(os.path.dirname(hdf5_file_path)):
        os.makedirs(os.path.dirname(hdf5_file_path))
    # print(hdf5_file_path)
    with h5py.File(hdf5_file_path, 'w') as f:
        
        data_group = f.create_group('data')

        mask_group = f.create_group('mask')
        train_names = np.array(["demo_{}".format(interact_index)], dtype='S')
        valid_names = np.array([], dtype='S')

        mask_group.create_dataset('train', data=train_names)
        mask_group.create_dataset('valid', data=valid_names)

        demo_group = data_group.create_group('demo_{}'.format(interact_index))
        obs_group = demo_group.create_group('obs')

    
        demo_group.attrs["num_samples"] = all_actions.shape[0]

        obs_group.create_dataset('proprios', data=all_proprios)
        obs_group.create_dataset('objects', data=all_objects)
        demo_group.create_dataset('actions', data=all_actions)
        demo_group.create_dataset('user_actions', data=all_user_actions)

        demo_group.create_dataset('corrects', data=all_corrects)
        
        print("{} saved".format(hdf5_file_path))
    
    all_proprios = []
    all_objects = []
    all_actions = []
    all_user_actions = []
    all_corrects = []
    if processes.index(current_process) < len(processes)-1:
        current_process = processes[processes.index(current_process)+1]


def change_gripper():
    global gripper
    global opened
    
    # if task == 'milk' and processes.index(current_process)==0:
    #     feedback = base_cyclic.RefreshFeedback()
    #     target = [feedback.base.tool_pose_x, feedback.base.tool_pose_y, feedback.base.tool_pose_z,
    #             feedback.base.tool_pose_theta_x,0, feedback.base.tool_pose_theta_z,]
    #     cartesian_action_movement(base, target)
        
    if gripper == 0 and not ((task=='drawer' or task == 'back') and current_process=='open'):
        save_file()
    elif gripper ==1:
        if ((task=='drawer' or task == 'back') and current_process=='open' and opened == True):
        # or (task=='cereal' and 'pour' in current_process and poured == True) :
            # save_file()
        
            success = int(input("Have you opened the drawer? Enter 0 or 1: "))
            if success ==1:
                save_file()
            else:
                opened = False
                open = 0.5
        elif task == 'back' and current_process=='open':
            assert not opened
            open = 0.5
    
    if gripper == 0:
        GC.ExampleSendGripperCommands(close)
    else:
        assert gripper == 1
        if (task == 'drawer' and processes.index(current_process)==len(processes)-1)\
        or (task == 'back' and opened):
            GC.ExampleSendGripperCommands(0.2)

        else:
            GC.ExampleSendGripperCommands(open)
            # print(open)

    gripper = 1 - gripper
    print('gripper:{}'.format(gripper))
    

def pour_process():
    global poured
    feedback = base_cyclic.RefreshFeedback()
    if not poured:
        poured = True
        global wrist_before_pour
        wrist_before_pour = feedback.actuators[5].position
        # while wrist_before_pour < -180:
        #     wrist_before_pour += 360
        # while wrist_before_pour > 180:
        #     wrist_before_pour -= 360
        
        # global relative_location
        current_y = feedback.base.tool_pose_y
        global y_before_pour
        y_before_pour = current_y
        # if preset_locations:
        #     cup_radius = 0.04
        #     bowl_radius = 0.075
        #     bowl_y = pour_targets[location_index][1] + \
        #     math.sin(math.radians(pour_targets[location_index][-1])) * (bowl_radius+cup_radius)
        # else:
        #     raise NotImplementedError
        # relative_location_calculated = 'l' if bowl_y > current_y else 'r'
        # relative_location = 'l' if bowl_y > pick_targets[location_index][1] else 'r'
        # print("bowl y: ", bowl_y)
        # print("current y: ", current_y)
        # print("relative location: ", relative_location)

        assert task == 'milk'
        global angles_before_poured
        angles_before_poured = np.array([feedback.base.tool_pose_theta_x, feedback.base.tool_pose_theta_z])
        global joints_before_pour
        joints_before_pour = np.array([feedback.actuators[0].position,
                                        feedback.actuators[1].position,
                                        feedback.actuators[2].position,
                                        feedback.actuators[3].position,
                                        feedback.actuators[4].position,
                                        feedback.actuators[5].position])
   
    if relative_location == 'l':
        send_joint_speeds(base, pour_speed)
    else:
        assert relative_location=='r'
        send_joint_speeds(base, -pour_speed)

    

def pull_drawer():
    global opened
    opened = True
    x_speed = - math.sin(math.radians(drawer_orientation)) * open_drawer_speed
    y_speed = + math.cos(math.radians(drawer_orientation)) * open_drawer_speed
    print("x_speed: ", x_speed)
    print("y_speed: ", x_speed)
    twist_command(base, [x_speed,y_speed,0,0,0,0])


    

def move(user_action, current_process):
    global gripper
    global correct
    global wrong

    global fight_seconds
    global wrong_start
    global obey_start
    global stop_start
    global fight_action
    global fight_stop_action

    global slow_start
    global fast_start


    significant_directions = (user_action[:3]>=xbox_threshold)
    
    proprios = get_proprios(task, loc_only, gripper)
    
    if poured:
        assert task == 'cereal'
        proprios[3:5] = angles_before_poured
        if not loc_only:
            proprios[5:11] = inverse_kinematics(base, [proprios[0],proprios[1],proprios[2],proprios[3],0,proprios[4]])
    
    proprios[3:-1][proprios[3:-1] > 180] -= 360
    proprios[3:-1][proprios[3:-1] < -180] += 360
    proprios_ori = proprios.copy()


    print(f"gripper: {gripper}")
    speeded_user_action = user_action.copy()
    # print(user_act)
    speeded_user_action[:3] *= move_speed
    if user_action.shape[-1]>3:
        speeded_user_action[3] *= move_speed * 100

    object_ori = object_ori_dict[current_process]
    if upgraded or latest:
        objects = Normalize(object_ori, min_objects, max_objects)
    else:
        objects = Normalize(object_ori, min_proprios[:object_ori.shape[0]], max_proprios[:object_ori.shape[0]])
    
    # print(proprios.shape)
    proprios[:-1] = Normalize(proprios[:-1], min_proprios[:(proprios.shape[-1]-1)], max_proprios[:(proprios.shape[-1]-1)])

    if user_action[-1] == 0:
        obs = {}                  
                   
        if transformer_enabled:           
            # print(proprios)
            assert len(proprios.shape)==len(objects.shape)==1
            if proprios_buffer.is_empty():
                assert objects_buffer.is_empty()                
                for t in range(frame_stack):
                    proprios_buffer.add(proprios)
                    objects_buffer.add(objects)
                
            else:
                proprios_buffer.add(proprios)
                objects_buffer.add(objects)
            assert len(proprios_buffer.buffer) == len(objects_buffer.buffer) == frame_stack
            
            proprios_list = proprios_buffer.get_all()
            objects_list = objects_buffer.get_all()
            
            proprios_all = np.stack(proprios_list)
            objects_all = np.stack(objects_list)
            
            assert len(proprios_all.shape)==len(objects_all.shape)==2 and proprios_all.shape[0]==objects_all.shape[0]==frame_stack
            
            obs['proprios'] = proprios_all
            # print(proprios_all.shape)
            # exit()
            obs['objects'] = objects_all
            obs['user_actions'] = np.array(user_action[:3])
            

        else:
            raise NotImplementedError
               
        for k in obs:
            obs[k] = obs[k].astype(np.float32).squeeze()           
        
        assert 'vanilla' not in ckpt_path
        assert finish_separate
        finish_act, middle_act, policy_act = policy(ob=obs) 
        policy_act_normalized = policy_act.copy()
        print("separate finish:", finish_act)
        
        # assert middle_act.shape[-1]>4
        # print("middle finish:", middle_act[-1])
        print("finish:", policy_act[-1])
        # middle_act = middle_act[:-1]
        policy_act = policy_act[:-1]
        

 
        speed_adjust = None
        if finish_by_distance:
            raise NotImplementedError #obj_ori_dict?
            distance = (math.sqrt((proprios_ori[0] - object_ori_dict[current_process][0])**2 + (proprios_ori[1] - object_ori_dict[current_process][1])**2 + (proprios_ori[2] - object_ori_dict[current_process][2])**2))
            maybe_stop = (distance <= stop_distance)
            maybe_slow = (distance > stop_distance and distance <= slow_distance)
            speed_computed_by_finish = (distance-slow_distance) /(stop_distance-slow_distance)
        else:
            maybe_stop = (finish_act[0] >= finish_upper_bound)
            maybe_slow = (finish_act[0]>=finish_lower_bound and finish_act[0]<finish_upper_bound)
            speed_computed_by_finish = (finish_upper_bound - finish_act[0]) /(finish_upper_bound-finish_lower_bound)
        maybe_stop = maybe_slow = False    
        if maybe_stop:
            if stop_start is None and fight_stop_action is None:
                stop_start = datetime.datetime.now()
                fight_stop_action = significant_directions
            if (significant_directions == fight_stop_action).all():
                if (datetime.datetime.now() - stop_start).total_seconds() < fight_stop_seconds:
                    base.Stop()
                    return
                else:
                    speed_adjust = finish_adjust_speed
                    print("speed adjust:", speed_adjust)
            else:
                stop_start = datetime.datetime.now()
                fight_stop_action = significant_directions
                base.Stop()
                return
        else:
            stop_start = None
            fight_stop_action = None

        
        policy_act = np.array(policy_act)
        policy_act = Unnormalize(policy_act, min_actions[:policy_act.shape[-1]], max_actions[:policy_act.shape[-1]])      
                
        assert len(policy_act.shape)==len(speeded_user_action.shape)==1 
        
        
        movement_scale = np.linalg.norm(policy_act[:3])
        print(f"movement scale:{movement_scale}")
        if scale:
            policy_act /= movement_scale/move_speed
            print(f"movement scaled to :{np.linalg.norm(policy_act[:3])}")
        else:
            policy_act *= speed
            print(f"movement speed:{np.linalg.norm(policy_act[:3])}")
            if not (maybe_slow or maybe_stop):
                adjust_speed = fast_speed
                if task == 'back' and np.linalg.norm(user_action[:3]) < 0.5 and np.linalg.norm(object_ori[:3] - proprios_ori[:3]) < 0.05:
                    adjust_speed *= 0.2
                if np.linalg.norm(policy_act[:3]) < adjust_speed:
                    policy_act[:3] *= adjust_speed/np.linalg.norm(policy_act[:3])
                    print(f"too slow, adjusting speed to:{np.linalg.norm(policy_act[:3])}")
    
    
        agreement = cosine_similarity(policy_act.copy()[:3], user_action[:3])
        print("unnormalized speeded policy action: ", policy_act)
        print("user action: ", user_action)
        print("agreement:{}".format(agreement))
        
        
        if agreement < agreement_threshold:
            if wrong_start is None and fight_action is None:
                wrong_start = datetime.datetime.now()
                fight_action = significant_directions
            if (significant_directions == fight_action).all():
                if (datetime.datetime.now() - wrong_start).total_seconds() < fight_seconds:
                    print("fight seconds: ", fight_seconds)
                    base.Stop()
                    return
                else:
                    if obey_start == None:
                        obey_start = datetime.datetime.now()
                    if (datetime.datetime.now() - obey_start).total_seconds() < obey_seconds:
                        wrong+=1
                        print("Using user action. The original user action is:", user_action)
                        action = speeded_user_action
                        all_corrects.append(0)
                    else:
                        print("fight seconds: ", fight_seconds)
                        base.Stop()
                        fight_seconds = fight_seconds_after_obey
                        wrong_start = datetime.datetime.now()
                        obey_start = None
                        return
            else:
                fight_seconds = fight_seconds_before_obey
                wrong_start = datetime.datetime.now()
                fight_action = significant_directions
                return

        else:   
            fight_seconds = fight_seconds_before_obey
            wrong_start = None
            obey_start = None
            fight_action = None
            correct+=1
            action = policy_act
            all_corrects.append(1)

          
        all_actions.append(policy_act_normalized)   

    else: #user enforces rotation
        raise NotImplementedError
        wrong+=1
        action = speeded_user_action
        assert (speeded_user_action[:3]==0).all()

        all_corrects.append(0)
        all_actions.append(np.zeros(action.shape))        


    all_proprios.append(proprios)         
    all_objects.append(objects)     
    all_user_actions.append(user_action)

    print("action:{}".format(action))
    

    #maybe adjust speed based on separate finish signal
    if user_action[-1] == 0:
        if maybe_stop:
            assert speed_adjust is not None
            action *= speed_adjust

        elif maybe_slow:
            # speed_computed_by_finish = (finish_upper_bound - finish_act[0]) /(finish_upper_bound-finish_lower_bound)
            speed_ori = np.linalg.norm(action[:3])/(0.005*speed)
            if speed_computed_by_finish<speed_ori:
                action *= speed_computed_by_finish/speed_ori
                print("speed computed by finish: ", speed_computed_by_finish)
                print("original speed: ", speed_ori)
                print(f"slowing down action by:{speed_computed_by_finish/speed_ori}")
    
    slow = np.linalg.norm(action[:3])<slow_speed
    if slow:
        if slow_start is None:
            slow_start = datetime.datetime.now()                     
        if (datetime.datetime.now() - slow_start).total_seconds() >= slow_time:
            if fast_start is None:
                fast_start = datetime.datetime.now()  
            if (datetime.datetime.now() - fast_start).total_seconds() < fast_time:   
                action *= fast_speed/np.linalg.norm(action[:3])
            else:
                slow_start = None
                fast_start = None
           
    else:
        slow_start = None
        fast_start = None

    
    print(f"action speed:{np.linalg.norm(action[:3])}")
    if task == 'pill':
        if processes.index(current_process)==0:
            if action.shape[-1]==5:
                print("(Using policy action)")
                twist_command(base, [action[0],action[1],action[2],0,0,action[4]])
            else:
                assert action[3]==0
                print("(Using user action)")
                twist_command(base, [action[0],action[1],action[2],0,0,0])
        elif processes.index(current_process)==1:
            twist_command(base, [action[0],action[1],action[2],action[3],0,0])
        else:
            twist_command(base, [action[0],action[1],action[2],0,0,0])

    elif task == 'cereal':
        print("Unnormalized action executed: ", action)
        if action.shape[-1]==5:
            print("(Using policy action)")
            twist_command(base, [action[0],action[1],action[2],0,action[3],action[4]])
        else:
            assert action[3]==0
            print("(Using user action)")
            twist_command(base, [action[0],action[1],action[2],action[3],0,0])
    else:
        raise NotImplementedError

        




    
                   
run_device = TorchUtils.get_torch_device(try_to_use_cuda=True)   

policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=run_device, verbose=True)
policy.start_episode()

all_proprios = []
all_objects = []
all_actions = []
all_user_actions = []
all_corrects = []

print(f"interact index:{interact_index}")

gripper = 0  
gripper_changed_time = None 
wrong_start = None 
obey_start = None
stop_start = None
fight_action = None
fight_stop_action = None
poured = False
opened = False

slow_start = None
fast_start = None


start_timing = None
correct = 0
wrong = 0

current_process = processes[0]

rotation = False
correct_num_dict = {}
wrong_num_dict = {}


with utilities.DeviceConnection.createTcpConnection(uargs) as router:       
    base = BaseClient(router)
    base_cyclic = BaseCyclicClient(router)

    actuator_count = base.GetActuatorCount().count   
    assert actuator_count == 6

    GC = GripperCommandExample(router, base)  
    GC.ExampleSendGripperCommands(open)

    zero_time = time.time()
    sampled_traj = []
    sampled_user_actions = []

    try:
        while True:

            feedback = base_cyclic.RefreshFeedback()
            sampled_traj.append([time.time() - zero_time, feedback.base.tool_pose_x, feedback.base.tool_pose_y, feedback.base.tool_pose_z])

            pygame.event.pump()

            # Listening to analog sticks
            left = -joystick.get_axis(0)
            forward = -joystick.get_axis(1)
            up = -joystick.get_axis(4)

            # print('xbox command:', left, forward, up)
            # continue
        
            # Listening to XYAB buttons
            put_down = joystick.get_button(0) #a
            pour = joystick.get_button(1) #b
            grasp = joystick.get_button(2) #x
            # rotate_up = joystick.get_button(3) #y

            next_mode = joystick.get_button(7) 

            if next_mode:

                rotation = ~rotation
                print(f"rotation: {rotation}")
                time.sleep(0.5)

            if sum([abs(left)>=0.25, abs(forward)>=0.25, abs(up)>=0.25, pour, grasp, put_down]) == 0:
                if task == 'cereal' and processes.index(current_process)==0:
                    # feedback = base_cyclic.RefreshFeedback()
                    adjust_degree = abs(feedback.base.tool_pose_theta_y)
                    current_theta_z = feedback.base.tool_pose_theta_z
                    target_theta_z = current_theta_z

                    
                    target = [feedback.base.tool_pose_x, feedback.base.tool_pose_y, feedback.base.tool_pose_z,
                            feedback.base.tool_pose_theta_x,0, target_theta_z]
                    cartesian_action_movement(base, target)
                base.Stop()
                continue
            
            # assert sum([abs(left)>=0.8, abs(forward)>=0.8, abs(up)>=0.8, rotate_down, pour, grasp, rotate_up]) == 1,\
            # (abs(left)>=0.8, abs(forward)>=0.8, abs(up)>=0.8, rotate_down, pour, grasp, rotate_up)
            # print((left, forward))
            # exit()
            if start_timing == None:
                start_timing = time.time()
            if time.time() - start_timing >=300:
                print("task failed")
                base.Stop()
                pygame.quit()
                exit()

            if grasp:
                if gripper_changed_time is None:
                    change_gripper()
                    gripper_changed_time = time.time()
                elif time.time() - gripper_changed_time > 1:
                    change_gripper()
                    gripper_changed_time = time.time()
            elif pour:
                if task == 'cereal':
                    pour_process()
            
     
            else:
                sampled_user_actions.append([forward, left, up])
                if rotation:
                    twist_command(base, [0,0,0,forward*10,left*10,up*10], mode='teleoperation')  
                else:
                    if (task == 'drawer' or task == 'back') and processes.index(current_process)==0 and gripper ==1:
                        pull_drawer()
                    else:
                        forward = forward if forward <= -xbox_threshold or forward > xbox_threshold else 0
                        left = left if left <= -xbox_threshold or left > xbox_threshold else 0
                        up = up if up <= -xbox_threshold or up > xbox_threshold else 0
                        user_action = np.array([forward, left, up, 0.])
                        

                        print('original user action:', user_action)               
                        move(user_action, current_process)
                        

    except KeyboardInterrupt:
        base.Stop()
        print('task completion time: ', time.time()-start_timing)

        sampled_traj = np.array(sampled_traj)
        sampled_user_actions = np.array(sampled_user_actions)
        save_folder = f'traj_stats/{npy_folder}/{task}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.save(f'{save_folder}/traj_{npy_name}.npy', sampled_traj)
        np.save(f'{save_folder}/user_actions_{npy_name}.npy', sampled_user_actions)
        print(f"Npy saved.")

        if processes.index(current_process)==len(processes)-1:
            if task == 'cereal':
                for p in range(processes.index(current_process)+1):
                    print(processes[p])
                    print(f"Correct actions: {correct_num_dict[processes[p]]}")
                    print(f"Wrong actions: {wrong_num_dict[processes[p]]}")
                    print(f"All actions: {correct_num_dict[processes[p]]+wrong_num_dict[processes[p]]}")
                print("object dict: ", object_ori_dict)
                if task == 'drawer' or task == 'back':
                    print("drawer orientation: ", drawer_orientation)
                # if task == 'trash' and rotate:
                if task == 'trash' or task == 'milk':
                    print("shelf orientation: ", shelf_orientation)
            else:
                save_file()      
        pygame.quit()
        

        


