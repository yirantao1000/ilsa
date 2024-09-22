import sys
import os
import threading
import time
import numpy as np
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.Exceptions.KServerException import KServerException
from kortex_api.autogen.messages import Base_pb2
from kortex_api.autogen.messages import BaseCyclic_pb2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities
uargs = utilities.parseConnectionArguments()
TIMEOUT_DURATION = 20

def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check


def send_joint_speeds(base, SPEED):

    joint_speeds = Base_pb2.JointSpeeds()
    
    del joint_speeds.joint_speeds[:]
    speeds = [0.0, 0.0, 0.0, 0.0, 0.0, SPEED]
    i = 0        
    for speed in speeds:
        joint_speed = joint_speeds.joint_speeds.add()
        joint_speed.joint_identifier = i 
        joint_speed.value = speed
        joint_speed.duration = 0
        i = i + 1
    
    base.SendJointSpeedsCommand(joint_speeds)


def twist_command(base, target, mode = 'shared'):

    command = Base_pb2.TwistCommand()

    if mode == 'shared':
        if target[4] == target[5]==0:
            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED
        else:
            command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
    else:
        assert mode == 'teleoperation'
        # target[3]=0
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_MIXED
    command.duration = 0

    twist = command.twist
    twist.linear_x = target[0]
    twist.linear_y = target[1]
    twist.linear_z = target[2]
    # print(twist.linear_z)
    twist.angular_x = target[3]
    twist.angular_y = target[4]
    twist.angular_z = target[5]

    base.SendTwistCommand(command)
    # time.sleep(1)
    # print('here')



def cartesian_action_movement(base, position):
    target = position
    
    # print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    # feedback = base_cyclic.RefreshFeedback()
    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = target[0]          # (meters)
    cartesian_pose.y = target[1]    # (meters)
    cartesian_pose.z = target[2]    # (meters)
    cartesian_pose.theta_x = target[3] # (degrees)
    cartesian_pose.theta_y = target[4] # (degrees)
    cartesian_pose.theta_z = target[5] # (degrees)

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    # print("Executing action")
    base.ExecuteAction(action)

    # print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    # if finished:
    #     print("Cartesian movement completed")
    # else:
    #     print("Timeout on action notification wait")

def angular_action_movement(base, target):
    
    # print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()

    # target = [356.011, 41.402, 212.176, 170.123, 289.166, 265.597]
    # target = [0.877740562, -0.012641931, 0.101506069, 98.50814819, 1.364000797, 92.32181549]
    # Place arm straight up
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        # joint_angle.value = math.radians(target[joint_id])

        joint_angle.value = target[joint_id]
        # joint_angle.value = 50

        if target[joint_id] > 147:
            joint_angle.value = target[joint_id] - 360
        else:
            joint_angle.value = target[joint_id]

        if abs(joint_angle.value)>147:
            joint_angle.value = 147 * ()
            
        # print(joint_id)
        # print(target[joint_id])
        # joint_angle.value = -120
    # print(action)
    # exit()
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    # print("Executing action")
    base.ExecuteAction(action)

    # print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    # if finished:
    #     print("Angular movement completed")
    # else:
    #     print("Timeout on action notification wait")
    return finished

class GripperCommandExample:
    def __init__(self, router, base, proportional_gain = 2.0):

        self.proportional_gain = proportional_gain
        self.router = router

        # Create base client using TCP router
        self.base = base

    def ExampleSendGripperCommands(self, position):

        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        # Close the gripper with position increments
        # print("Performing gripper test in position...")
        gripper_command.mode = Base_pb2.GRIPPER_POSITION
        # position = 0.00
        finger.finger_identifier = 1
        # while position < 1.0:
        #     finger.value = position
        #     print("Going to position {:0.2f}...".format(finger.value))
        #     self.base.SendGripperCommand(gripper_command)
        #     position += 0.1
        #     time.sleep(1)
        finger.value = position
        # print("Going to position {:0.2f}...".format(finger.value))
        self.base.SendGripperCommand(gripper_command)
        time.sleep(0.5)


class GripperLowLevelExample:
    def __init__(self, router, router_real_time, base, base_cyclic, proportional_gain = 2.0):

        self.proportional_gain = proportional_gain

        self.router = router
        self.router_real_time = router_real_time

        # Create base client using TCP router
        # self.base = BaseClient(self.router)
        self.base = base

        # Create base cyclic client using UDP router.
        self.base_cyclic = BaseCyclicClient(self.router_real_time)
        # self.base_cyclic = base_cyclic

        # Create base cyclic command object.
        self.base_command = BaseCyclic_pb2.Command()
        self.base_command.frame_id = 0
        self.base_command.interconnect.command_id.identifier = 0
        self.base_command.interconnect.gripper_command.command_id.identifier = 0

        # Add motor command to interconnect's cyclic
        self.motorcmd = self.base_command.interconnect.gripper_command.motor_cmd.add()

        # Set gripper's initial position velocity and force
        base_feedback = self.base_cyclic.RefreshFeedback()
        self.motorcmd.position = base_feedback.interconnect.gripper_feedback.motor[0].position
        self.motorcmd.velocity = 0
        self.motorcmd.force = 100

        for actuator in base_feedback.actuators:
            self.actuator_command = self.base_command.actuators.add()
            self.actuator_command.position = actuator.position
            self.actuator_command.velocity = 0.0
            self.actuator_command.torque_joint = 0.0
            self.actuator_command.command_id = 0
            # print("Position = ", actuator.position)

        # Save servoing mode before changing it
        self.previous_servoing_mode = self.base.GetServoingMode()

        # Set base in low level servoing mode
        self.servoing_mode_info = Base_pb2.ServoingModeInformation()
        self.servoing_mode_info.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
    


    def Goto(self, target_position):
        self.base.SetServoingMode(self.servoing_mode_info)
        time.sleep(0.1)
        
        while True:
            try:
                base_feedback = self.base_cyclic.Refresh(self.base_command)

                # Calculate speed according to position error (target position VS current position)
                position_error = target_position - base_feedback.interconnect.gripper_feedback.motor[0].position

                # If positional error is small, stop gripper
                if abs(position_error) < 1.5:
                    position_error = 0
                    self.motorcmd.velocity = 0
                    self.base_cyclic.Refresh(self.base_command)
                    return True
                else:
                    self.motorcmd.velocity = self.proportional_gain * abs(position_error)
                    if self.motorcmd.velocity > 100.0:
                        self.motorcmd.velocity = 100.0
                    self.motorcmd.position = target_position

            except Exception as e:
                print("Error in refresh: " + str(e))
                return False
            time.sleep(0.001)
            self.base.SetServoingMode(self.previous_servoing_mode)
        return True
    
    # def Goto(self, target_position):
    #     # self.base.SetServoingMode(self.servoing_mode_info)
    #     # time.sleep(1)

    #     # self.base_cyclic.Refresh(self.base_command)
        
    #     # time.sleep(0.1)
        
    #     self.motorcmd.velocity = 100
    #     self.motorcmd.position = target_position
    #     # time.sleep(0.1)
    #     # self.base.SetServoingMode(self.previous_servoing_mode)
       

    #     # base_feedback = self.base_cyclic.Refresh(self.base_command)

    #     #         # Calculate speed according to position error (target position VS current position)
    #     #         position_error = target_position - base_feedback.interconnect.gripper_feedback.motor[0].position

    #     #         # If positional error is small, stop gripper
    #     #         self.motorcmd.velocity = self.proportional_gain * abs(position_error)
    #     #         if self.motorcmd.velocity > 100.0:
    #     #             self.motorcmd.velocity = 100.0
    #     #         self.motorcmd.position = target_position

    #     #     except Exception as e:
    #     #         print("Error in refresh: " + str(e))
    #     #         return False
    #     #     time.sleep(1)
    #     #     self.base.SetServoingMode(self.previous_servoing_mode)
    #     # return True

def inverse_kinematics(base, target):

    try:
        input_joint_angles = base.GetMeasuredJointAngles()
    except KServerException as ex:
        print("Unable to get current robot pose")
        print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
        print("Caught expected error: {}".format(ex))
        raise NotImplementedError
    
    # Object containing cartesian coordinates and Angle Guess
    input_IkData = Base_pb2.IKData()
    
    # Fill the IKData Object with the cartesian coordinates that need to be converted
    input_IkData.cartesian_pose.x = target[0]
    input_IkData.cartesian_pose.y = target[1]
    input_IkData.cartesian_pose.z = target[2]
    input_IkData.cartesian_pose.theta_x = target[3]
    input_IkData.cartesian_pose.theta_y = target[4]
    input_IkData.cartesian_pose.theta_z = target[5]

    # Fill the IKData Object with the guessed joint angles
    for joint_angle in input_joint_angles.joint_angles :
        jAngle = input_IkData.guess.joint_angles.add()
        # '- 1' to generate an actual "guess" for current joint angles
        # jAngle.value = joint_angle.value - 1
        jAngle.value = joint_angle.value
    
    try:
        # print("Computing Inverse Kinematics using joint angles and pose...")
        computed_joint_angles = base.ComputeInverseKinematics(input_IkData)
    except KServerException as ex:
        print("Unable to compute inverse kinematics")
        print("Error_code:{} , Sub_error_code:{} ".format(ex.get_error_code(), ex.get_error_sub_code()))
        print("Caught expected error: {}".format(ex))
        raise NotImplementedError

    # print("Joint ID : Joint Angle")
    joint_identifier = 0
    results = []
    for joint_angle in computed_joint_angles.joint_angles :
        # print(joint_identifier, " : ", joint_angle.value)
        results.append(joint_angle.value)
        joint_identifier += 1

    return np.array(results)