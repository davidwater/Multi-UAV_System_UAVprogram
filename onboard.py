#!/usr/bin/env python3

from digi.xbee.devices import XBee64BitAddress, DigiMeshDevice, RemoteDigiMeshDevice
from drone import *
from communication_info import *
from time import time
import rospy
import multiprocessing as mp
import DPGA
from two_UAV_SDPSO_main import *
import signal
import sys
t = time.time

def signal_handler(signal, frame):
    sys.exit(0)

class Timer(object):
    def __init__(self):
        self.bias = None
        self.t1, self.t2, self.t3, self.t4 = None, None, None, None
        self.interval = 2
        self.delay = None

    def t(self):
        sync = False
        if int((t() + self.bias + self.delay) * 10) % int(self.interval * 10) == 0:
            print(f'time sychronization: {int((t() + self.bias + self.delay) * 10) % int(self.interval * 10) == 0}')
            sync = True
        return t() + self.bias, sync

    def check_timer(self, interval, previous_send_time, delay=0):
        '''
        sycronized communication (metrics: transmission frequency)
        '''
        if int((t() + self.bias + delay) * 10) % int(interval * 10) == 0 and t() - previous_send_time >= interval/1.01:
            return True
        else:
            return False
        
    def check_deciTime(self, deciTime):
        if int((t() + self.bias) * 10 % 10) == deciTime:
            return True
        else:
            return False

    def check_period(self, period, previous_time):
        if t() - previous_time >= period:
            return True
        else:
            return False

    def time_synchronize_process(self, central_device, client, client_id):
        '''
        Cristian algorithm: theta = (t2 - t1 + t3 - t4)/2
        '''
        self.bias = None
        while not self.bias:
            print("time sycronization request")
            client.send_data_async(central_device, bytearray([Message_ID.Time_Synchromize.value, client_id]))
            self.t1 = t()

            while t() - self.t1 < 5:
                try:
                    packet = client.read_data(1e-5)
                    if packet.data[0] == Message_ID.Time_Synchromize.value and packet.data[1] == client_id:
                        self.t4 = packet.timestamp
                        t2, t3 = unpack('dd', packet.data[2:])
                        self.t2 = t2
                        self.t3 = t3
                        self.bias = (self.t2 - self.t1 + self.t3 - self.t4)/2
                        self.delay = ((self.t4 - self.t1) - (self.t3 - self.t2))/2
                        print(f"Bias: {self.bias}, t1:{self.t1}, t2:{self.t2}, t3:{self.t3}, t4:{self.t4}, delay:{((self.t4 - self.t1) - (self.t3 - self.t2))/2}")
                        # print(f'time sychronization: {int((t() + self.bias + self.delay) * 10) % int(self.interval * 10) == 0}')
                        break
                except:
                    continue


if __name__ == "__main__":
    ' <<<<<<<<<<<<<<<<<<<<<   Initialization   >>>>>>>>>>>>>>>>>>>>>>> '
    ' => XBee connection '
    xbee = DigiMeshDevice('/dev/ttyUSB0', 115200)
    xbee.open(force_settings=True)
    uav_id = int(xbee.get_node_id())

    ' => ROS connection '
    rospy.init_node('drone', anonymous=True)    
    UAV = Drone()
    sdpso = SDPSO(np.matrix([0,0,0,0]), np.matrix([0,0,0,0]), np.matrix([0,0,0,0]), np.matrix([0,0]), np.array([0,0]), np.array([0,0]), it = 0, ds = 10, Varsize = 4)
    
    ' => Communication setting '
    data = packet_processing(uav_id)
    u2u_address = [RemoteDigiMeshDevice(xbee, XBee64BitAddress.from_hex_string(xb.value)) for xb in XBee_Devices if not xb.name == f"UAV{uav_id}"]
    gcs_address = RemoteDigiMeshDevice(xbee, XBee64BitAddress.from_hex_string("0013A2004127B732"))  # X2 ConnectPort
    u2u_interval = 2
    u2g_interval = 0.5
    previous_u2u_time, previous_u2g_time = 0, 0

    ' => Time calibration'
    new_timer = Timer()
    new_timer.time_synchronize_process(gcs_address, xbee, uav_id)

    ' => Mission ' 
    Mission = Message_ID.Default
    stop_mode = [Mode.LAND.name, Mode.RTL.name, Mode.LOITER.name]
    target, index, waypoint_radius = [], 0, 5
    completed = False
    previous_cmd_time = 0
    
    ' <<<<<<<<<<<<<<<<<<<<<   MainProgram   >>>>>>>>>>>>>>>>>>>>>>> '
    while not rospy.is_shutdown():
        ' receive data (U2U)(G2U)'
        try:
            packet = xbee.read_data(timeout=1e-5)
            if packet:
                messageType, info = data.unpack_packet(packet.data)
                print(messageType, info, new_timer.t())
                if messageType == Message_ID.Mode_Change:
                    success = UAV.set_mode(info.name)

                elif messageType == Message_ID.info:
                    xbee.send_data_async(gcs_address, data.pack_info_packet(info))

                elif messageType == Message_ID.Arm:
                    if info == Armed.armed:
                        if not UAV.armed:
                            UAV.set_arm()
                        else:
                            xbee.send_data_async(gcs_address, data.pack_info_packet(f"has already armed!"))
                    elif info == Armed.disarmed:
                        if UAV.armed:
                            UAV.set_disarm()
                        else:
                            xbee.send_data_async(gcs_address, data.pack_info_packet(f"has already disarmed!"))
                
                elif messageType == Message_ID.Time_Synchromize:
                    new_timer.time_synchronize_process(gcs_address, xbee, uav_id)

                elif messageType == Message_ID.Takeoff:
                    UAV.set_mode(Mode.GUIDED.name)
                    UAV.set_arm()
                    success = UAV.takeoff(info)
                    if success:
                        xbee.send_data_async(gcs_address, data.pack_info_packet(f"perform Takeoff ({info}m)"))
                    else:
                        xbee.send_data_async(gcs_address, data.pack_info_packet(f"fail to Takeoff ({info}m)"))

                elif messageType == Message_ID.Comm_u2gFreq:
                    u2g_interval = 1/info
                    xbee.send_data_async(gcs_address, data.pack_info_packet(f"U2G frquency: {info}Hz"))

                elif messageType == Message_ID.Origin_Correction:
                    UAV.origin_correction(info)
                    xbee.send_data_async(gcs_address, data.pack_info_packet(f"origin changed => {info}"))   

                elif messageType == Message_ID.Waypoints:
                    method = info[0]
                    xbee.send_data_async(gcs_address, data.pack_info_packet(f"received Waypoints command"))
                    if not UAV.mode == Mode.GUIDED.name:
                        success = UAV.set_mode(Mode.GUIDED.name)

                    if method == WaypointMissionMethod.guide_waypoint:
                        Mission = Message_ID.Waypoints
                        waypoint_radius = info[1]
                        target = info[2]
                        target[2] = np.round(UAV.local_pose[2]) if target[2] == 0 else target[2]
                        completed = False

                    elif method == WaypointMissionMethod.guide_WPwithHeading:
                        Mission = Message_ID.Waypoints
                        waypoint_radius = info[1]
                        target = info[2]
                        target[2] = np.round(UAV.local_pose[2]) if target[2] == 0 else target[2]
                        completed = False

                    elif method == WaypointMissionMethod.guide_waypoints:
                        Mission = Message_ID.Waypoints
                        waypoint_radius = info[1]
                        target = info[2]
                        index = 0
                        completed = False

                    elif method == WaypointMissionMethod.CraigReynolds_Path_Following:
                        Mission = Message_ID.Waypoints
                        waypoint_radius = info[1]
                        CRPF = info[2]
                        UAV.v = info[-1][0]
                        UAV.Rmin = info[-1][1]
                        index = 0
                        completed = False
                        pre_error = None
                        height = np.round(UAV.local_pose[2])

                elif messageType == Message_ID.SEAD_mission:
                    height = np.round(UAV.local_pose[2])
                    Mission = Message_ID.SEAD_mission
                    xbee.send_data_async(gcs_address, data.pack_record_time_packet(f"received SEAD mission", new_timer.t()))
                    taskAllocation2main, main2taskAllocation = mp.Queue(), mp.Queue()
                    taskAllocationProcess = mp.Process(target=DPGA.task_allocation_process, args=(info[0], 0.5, 100, taskAllocation2main, main2taskAllocation))
                    mainProcess = DPGA.main_process(info[0], info[1], info[3], u2u_address, taskAllocation2main, main2taskAllocation)
                    if not UAV.armed:  # sim UAV
                        print("sim UAV activate")
                        UAV = DPGA.UAV_Simulator(uav_id, info[4][0], FrameType.Quad, info[4][1], info[4][2], info[2], info[3])
                    UAV.type = info[4][0]
                    UAV.v = info[4][1]
                    UAV.Rmin = info[4][2]
                    waypoint_radius = info[-1]
                    taskAllocationProcess.start()
                    if not UAV.mode == Mode.GUIDED.name:
                        success = UAV.set_mode(Mode.GUIDED.name)

                elif messageType == Message_ID.SDPSO:
                    height = np.round(UAV.local_pose[2])
                    Mission = Message_ID.SDPSO
                    back_to_base = False
                    completed = False       
                    pre_error = None
                    index = 0
                    update = True
                    xbee.send_data_async(gcs_address, data.pack_record_time_packet(f"received SDPSO mission", new_timer.t()))
                    UAV.Rmin = info[0,0]
                    if uav_id == 1:
                        sdpso.start[0,0:2] = np.array([[info[1][0,0], info[1][0,1]]])
                        sdpso.target[0,0:2] = np.array([[info[1][1,0], info[1][1,1]]])
                    elif uav_id ==  2:
                        sdpso.start[0,2:4] = np.array([[info[1][0,0], info[1][0,1]]])
                        sdpso.target[0,2:4] = np.array([[info[1][1,0], info[1][1,1]]])         
                
                elif messageType == Message_ID.Mission_Abort:
                    Mission = Message_ID.Mission_Abort
                    xbee.send_data_async(gcs_address, data.pack_info_packet(f"mission stop and hold"))
        except:
            pass
        
        ' Send data to GCS (U2G) '
        if new_timer.check_timer(u2g_interval, previous_u2g_time):  
            previous_u2g_time = t()
            try:
                send_packet = data.pack_u2g_packet_default(Mission, UAV.frame_type, UAV.mode, UAV.armed, UAV.battery_perc, 
                                                           new_timer.t(), UAV.local_pose, UAV.roll, UAV.pitch, UAV.yaw, 
                                                           np.linalg.norm(UAV.local_velo))
                xbee.send_data_async(gcs_address, send_packet)
            except:
                pass
        
        ' Mission program '
        if Mission == Message_ID.Waypoints:
            if new_timer.check_period(0.1, previous_cmd_time):
                previous_cmd_time = t()
                if method == WaypointMissionMethod.guide_waypoint:
                    UAV.guide_to_waypoint(target)
                    if np.linalg.norm(target - np.array(UAV.local_pose)) <= waypoint_radius and not completed:
                        xbee.send_data_async(gcs_address, data.pack_info_packet(f"arrive at {target}"))
                        completed = True

                elif method == WaypointMissionMethod.guide_WPwithHeading:
                    UAV.guide_to_waypoint(target[:3], target[-1]*pi/180)
                    if np.linalg.norm(target[:3] - np.array(UAV.local_pose)) <= waypoint_radius and not completed:
                        xbee.send_data_async(gcs_address, data.pack_info_packet(f"arrive at {target[:3]} with {target[-1]} deg"))
                        completed = True

                elif method == WaypointMissionMethod.guide_waypoints:
                    UAV.guide_to_waypoint(target[index])
                    if np.linalg.norm(target[index] - np.array(UAV.local_pose)) <= waypoint_radius and not completed:
                        index += 1
                        if index == len(target):
                            index = -1
                            xbee.send_data_async(gcs_address, data.pack_info_packet("Waypoints mission completed!"))
                            completed = True
                
                elif method == WaypointMissionMethod.CraigReynolds_Path_Following:
                    if CRPF.method == pathFollowingMethod.path_following_position:
                        desirePoint, index, _ = CRPF.get_desirePoint(UAV.v, UAV.local_pose[0], UAV.local_pose[1], UAV.yaw)
                        UAV.guide_to_waypoint([desirePoint[0], desirePoint[1], CRPF.path[index][-1]])

                    elif CRPF.method == pathFollowingMethod.path_following_position_yaw:
                        desirePoint, index, _ = CRPF.get_desirePoint(UAV.v, UAV.local_pose[0], UAV.local_pose[1], UAV.yaw)
                        UAV.guide_to_waypoint([desirePoint[0], desirePoint[1], CRPF.path[index][-1]], arctan2(desirePoint[1] - UAV.local_pose[1], desirePoint[0] - UAV.local_pose[0])) 

                    elif CRPF.method == pathFollowingMethod.path_following_velocityLocal:
                        if not completed:
                            UAV.velocity_control(CRPF.get_desireVelocity(UAV.v, UAV.local_pose[0], UAV.local_pose[1], UAV.local_velo[0], UAV.local_velo[1])[0])
                        else:
                            UAV.velocity_control(0, 0, 0)

                    elif CRPF.method == pathFollowingMethod.dubinsPath_following_velocityBody_PID:
                        if UAV.frame_type == FrameType.Quad:
                            desirePoint, index, _, error_of_distance = CRPF.get_desirePoint_withWindow(UAV.v, UAV.local_pose[0], UAV.local_pose[1], UAV.yaw, index)
                            u, pre_error = CRPF.PID_control(UAV.v, UAV.Rmin, UAV.local_pose, UAV.yaw, desirePoint, pre_error)
                            if error_of_distance <= 0 and completed:
                                target_V, u = 0, 0
                            else:
                                target_V = UAV.v
                            v_z = 0.3 * (height - UAV.local_pose[2])  # altitude hold
                            UAV.velocity_bodyFrame_control(target_V, u, v_z)
                        elif UAV.frame_type == FrameType.Fixed_wing:
                            desirePoint, index, _, error_of_distance = CRPF.get_desirePoint_withWindow(UAV.v, UAV.local_pose[0], UAV.local_pose[1], UAV.yaw, index)
                            if error_of_distance <= 0 and completed:
                                UAV.set_mode(Mode.LOITER.name)
                            else:
                                UAV.position_control(desirePoint[0], desirePoint[1], height)
                
                    if np.linalg.norm(CRPF.path[-1][:2] - np.array(UAV.local_pose[:2])) <= waypoint_radius and not completed:
                        xbee.send_data_async(gcs_address, data.pack_record_time_packet(f"CraigReynolds Path Following {CRPF.method.name} mission completed!", new_timer.t()))
                        completed = True
                       
        elif Mission == Message_ID.SEAD_mission:
            if type(UAV) == Drone:
                if UAV.frame_type == FrameType.Quad:
                    mainProcess.run_quadcopter(xbee, data, UAV, new_timer, gcs_address, height, waypoint_radius)
                elif UAV.frame_type == FrameType.Fixed_wing:
                    mainProcess.run_fixedWing(xbee, data, UAV, new_timer, gcs_address, height, waypoint_radius)
            else:
                mainProcess.run_simulation(xbee, data, UAV, new_timer, gcs_address, waypoint_radius)

        elif Mission == Message_ID.SDPSO:
            data_u2u = packet_processing(uav_id)
            previous_time_u2u = 0
            if uav_id == 1:
                sdpso.start[0,2:4] = np.array([-150,100]) 
                sdpso.target[0,2:4] = np.array([-200,10])  
            elif uav_id == 2:
                sdpso.start[0,0:2] = np.array([-200,10])
                sdpso.target[0,0:2] = np.array([-150,100]) 
            ' Broadcast every T seceods'
            _, sync = new_timer.t()
            while sync == False:
                _, sync = new_timer.t()
                if sync == True:
                    break
            
            if uav_id ==1:
                if new_timer.check_timer(u2u_interval, previous_time_u2u, delay = -0.1) and not back_to_base:
                    print('ok')
                    previous_time_u2u = time.time()
                    UAV1_packet = data_u2u.pack_SDPSO_packet(sdpso.start, sdpso.target)
                    xbee.send_data_broadcast(UAV1_packet)

                    # Receive the information of UAVs after Tcomm seconds
                    if new_timer.check_period(0.5, previous_time_u2u) and UAV1_packet:
                        if update:
                            sdpso.start[0,2:4] = data_u2u.uavs_info[1]
                            sdpso.target[0,2:4] = data_u2u.uavs_info[2]
                            update = False
                            print('data exchange!')
                        else:
                            print('no data to exchange')
            
            if uav_id ==2:
                print(new_timer.check_timer(u2u_interval, previous_time_u2u, delay = -0.1))
                print(not back_to_base)
                if new_timer.check_timer(u2u_interval, previous_time_u2u, delay = -0.1) and not back_to_base:
                    print('ok')
                    previous_time_u2u = time.time()
                    UAV2_packet = data_u2u.pack_SDPSO_packet(sdpso.start, sdpso.target)
                    xbee.send_data_broadcast(UAV2_packet)

                    # Receive the information of UAVs after Tcomm seconds
                    if new_timer.check_period(0.5, previous_time_u2u) and UAV2_packet:
                        if update:
                            sdpso.start[0,0:2] = data_u2u.uavs_info[1]
                            sdpso.target[0,0:2] = data_u2u.uavs_info[2]
                            update = False
                            print('data exchange!')
                        else:
                            print('no data to exchange')

            v = np.matrix([0,0,0,0])
            path_1, path_2, h, d_total, cost = generate_path(sdpso.start, sdpso.target, v)
            path_1, path_2 = smooth_path(path_1, path_2)
            UAV.v = 3
            print('SDPSO iteration finish')
            xbee.send_data_async(gcs_address, data.pack_record_time_packet(f"UAV{uav_id} SDPSO iteration finish!", new_timer.t()))

            if uav_id == 1:
                tracking1 = CraigReynolds_Path_Following(WaypointMissionMethod.CraigReynolds_Path_Following, 1, path = path_1, path_window = 3, Kp = 1, Kd = 5)
                desirePoint, index, _, error_of_distance = tracking1.get_desirePoint_withWindow(UAV.v, UAV.local_pose[0], UAV.local_pose[1], UAV.yaw, index)
                u, pre_error = tracking1.PID_control(UAV.v, UAV.Rmin, UAV.local_pose, UAV.yaw, desirePoint, pre_error)
                if error_of_distance <= 0 and completed:
                    target_V, u = 0, 0
                    UAV.set_mode(Mode.POSHOLD.name)
                    xbee.send_data_async(gcs_address, data.pack_record_time_packet(f"UAV1 set to POSHOLD!", new_timer.t()))
                else:
                    target_V = UAV.v
                v_z = 0.3 * (height - UAV.local_pose[2])  # altitude hold
                UAV.velocity_bodyFrame_control(target_V, u, v_z)

                if np.linalg.norm(path_1[-1][:2] - np.array(UAV.local_pose[:2])) <= waypoint_radius and not completed:
                    xbee.send_data_async(gcs_address, data.pack_record_time_packet(f"UAV1 SDPSO mission completed!", new_timer.t()))
                    completed = True
                    back_to_base = True

            elif uav_id == 2:
                tracking2 = CraigReynolds_Path_Following(WaypointMissionMethod.CraigReynolds_Path_Following, 1, path = path_2, path_window = 3, Kp = 1, Kd = 5)
                desirePoint, index, _, error_of_distance = tracking2.get_desirePoint_withWindow(UAV.v, UAV.local_pose[0], UAV.local_pose[1], UAV.yaw, index)
                u, pre_error = tracking2.PID_control(UAV.v, UAV.Rmin, UAV.local_pose, UAV.yaw, desirePoint, pre_error)
                if error_of_distance <= 0 and completed:
                    target_V, u = 0, 0
                    UAV.set_mode(Mode.POSHOLD.name)
                    xbee.send_data_async(gcs_address, data.pack_record_time_packet(f"UAV2 set to POSHOLD!", new_timer.t()))
                else:
                    target_V = UAV.v
                v_z = 0.3 * (height - UAV.local_pose[2])  # altitude hold
                UAV.velocity_bodyFrame_control(target_V, u, v_z)

                if np.linalg.norm(path_2[-1][:2] - np.array(UAV.local_pose[:2])) <= waypoint_radius and not completed:
                    xbee.send_data_async(gcs_address, data.pack_record_time_packet(f"UAV2 SDPSO mission completed!", new_timer.t()))
                    completed = True
            


        elif Mission == Message_ID.Mission_Abort:
            if UAV.frame_type == FrameType.Quad:
                UAV.velocity_control([0, 0, 0])
            elif UAV.frame_type == FrameType.Fixed_wing:
                UAV.set_mode(Mode.LOITER.name)

            
        ' Mission cancal mechanism '
        if UAV.mode in stop_mode:
            Mission = Message_ID.Default
    signal.signal(signal.SIGINT, signal_handler)
    xbee.send_data_async(gcs_address, data.pack_info_packet(f"rospy is shutdown!!"))
 
