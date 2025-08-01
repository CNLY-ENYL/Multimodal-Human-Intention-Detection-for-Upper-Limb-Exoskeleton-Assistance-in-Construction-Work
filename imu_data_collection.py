
if __name__ == "__main__" :
    print("\033cStarting ...\n") # Clear Terminal

# ----   # Modifiable variables   ----
root_directory: str =   'Temporary_Data'    # Directory where temporary folders are stored
Ask_cam_num: bool =     False               # Set to True to ask the user to put the cam number themselves, if False, default is set below
cam_num: int =          1                   # Set to 0 to activate the camera, but 1 if yoy have a builtin camera
NEW_CAM : bool =        False               # Set to True if you are using the new camera
fps: int =              30                  # Number of save per seconds
buffer: int =           1500                # Number of folders saved
CleanFolder: bool =     False               # If True, delete all temporary folders at the end
wifi_to_connect: str =  'Upper_Limb_Exo'    # The Wi-Fi where the raspberry pi and IMUs are connected
window_size: int =      200                  # How many lines of IMU data will be displayed at the same time
PRINT_IMU =             True                # If true print the imu data in the terminal
IMU_IDS = ['68592362', '68591F90', '685928A2', '68592647', '685920B2']
# IMU:  LEH, LSE, LB, REH, RES
# IMU X-DIR: L H<-E->S, B H->F, R H<-E->S
# ------------------------------------


import csv                      # For csv writing
import os                       # To manage folders and paths
import sys                      # For quitting program early
import math                     # To calculate elbow angle(in Print function)
import threading
import keyboard
import pandas as pd
import numpy as np
from time import sleep, time

try :
    import cv2      # For the camera
    import ximu3    # For the IMU
    from pupil_labs.realtime_api.simple import discover_one_device
    import pandas as pd
except ModuleNotFoundError as Err :
    missing_module = str(Err).replace('No module named ','')
    missing_module = missing_module.replace("'",'')
    if missing_module == 'cv2' :
        sys.exit(f'No module named {missing_module} try : pip install opencv-python')
    elif missing_module == "pupil_labs" :
        sys.exit(f'No module named {missing_module} try : pip install pupil-labs-realtime-api')
    else :
        print(f'No module named {missing_module} try : pip install {missing_module}')

try :
    from Imports.Functions import format_time, connected_wifi, ask_yn, calculate_angle_between_vectors, extract_arm_direction_angles, process_imu_to_new
except ModuleNotFoundError :
    sys.exit('Missing Import folder, make sure you are in the right directory')

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

try :
    if Ask_cam_num :
        NEW_CAM = ask_yn('\033cAre you using the new camera ?(Y/N) ')
        if  not NEW_CAM:
            cam_num = int(input("Cam Number : "))
        if cam_num < 0: 
            raise ValueError
except (ValueError, TypeError) :
    sys.exit("Invalid Cam Number") 
except KeyboardInterrupt :
    sys.exit("\n\nProgramme Stopped\n")

# We check if the root directory exist
if not os.path.exists(root_directory) :
    os.makedirs(root_directory)
elif os.listdir(root_directory):  # If there are files in the directory : True
    if ask_yn(f'\033c{root_directory} not empty do you want to clear it ? (Y/N)') :
        print('Clearing ...')
        for folders_to_del in os.listdir(root_directory):
            for files_to_del in os.listdir(f"{root_directory}/{folders_to_del}"):
                os.remove(os.path.join(f'{root_directory}/{folders_to_del}', files_to_del))
            os.rmdir(f"{root_directory}/{folders_to_del}")
    elif ask_yn('Do you want to save it ? (Y/N)') :
        Folder_Name = str(input("Folder Name : "))
        if root_directory != Folder_Name and Folder_Name != '' :
            os.rename(root_directory, Folder_Name)
        else : sys.exit("Incorrect Folder Name")
    else : sys.exit('Cannot access non-empty folder, Programme Stopped\n')

print("\033cStarting ...\n") # Clear Terminal
print("Checking Wifi ...")

ConnectedWifi = connected_wifi()
if ConnectedWifi[0] :
    if ConnectedWifi[1] != wifi_to_connect and ConnectedWifi[1] != wifi_to_connect+'_5G' :
        sys.exit('Not connected to the right wifi')
    else : 
        print(LINE_UP, end=LINE_CLEAR)
        print(f'Connected to {ConnectedWifi[1]}')
else : print("Could not check Wifi")



# Initialize sensor values to 0
imu_data = {
    imu_id: {
        "gyr": [0, 0, 0],
        "acc": [0, 0, 0],
        "rotm": [[0.0] * 3 for _ in range(3)]
    } for imu_id in IMU_IDS
}

is_paused = False  # press "space" button then pause  the datacollection

class Connection:
    def __init__(self, connection_info):
        self.__connection = ximu3.Connection(connection_info)
        if self.__connection.open() != ximu3.RESULT_OK:
            sys.exit("Unable to open connection " + connection_info.to_string())
        ping_response = self.__connection.ping()
        self.__prefix = ping_response.serial_number
        if ping_response.result != ximu3.RESULT_OK:
            print("Ping failed for " + connection_info.to_string())
            raise AssertionError
        self.__connection.add_inertial_callback(self.__inertial_callback)
        self.__connection.add_rotation_matrix_callback(self.__rotation_matrix_callback)

    def close(self):
        self.__connection.close()

    def send_command(self, key, value=None):
        if value is None:
            value = "null"
        elif type(value) is bool:
            value = str(value).lower()
        elif type(value) is str:
            value = "\"" + value + "\""
        else:
            value = str(value)

        command = "{\"" + key + "\":" + value + "}"

        responses = self.__connection.send_commands([command], 2, 500)

        if not responses:
            sys.exit("Unable to confirm command " + command + " for " + self.__connection.get_info().to_string())
        else:
            print(self.__prefix + " " + responses[0])

    def __inertial_callback(self, msg):
        if self.__prefix in imu_data:
            imu_data[self.__prefix]["gyr"] = [msg.gyroscope_x, msg.gyroscope_y, msg.gyroscope_z]
            imu_data[self.__prefix]["acc"] = [msg.accelerometer_x, msg.accelerometer_y, msg.accelerometer_z]

    def __rotation_matrix_callback(self, msg):
        if self.__prefix in imu_data:
            imu_data[self.__prefix]["rotm"] = [
                [msg.xx, msg.xy, msg.xz],
                [msg.yx, msg.yy, msg.yz],
                [msg.zx, msg.zy, msg.zz]
            ]

def key_listener():
    global is_paused
    while True:
        keyboard.wait('space')
        is_paused = not is_paused
        print("\n[PAUSED]" if is_paused else "\n[RESUMED]")



# Establish connections
print("Checking IMU connections...")
while True:
    try:
        connections = [Connection(m.to_udp_connection_info()) for m in ximu3.NetworkAnnouncement().get_messages_after_short_delay()]
        break
    except AssertionError:
        pass
if not connections:
    print(LINE_UP, end=LINE_CLEAR)
    sys.exit("No UDP connections to IMUs")
print(LINE_UP, end=LINE_CLEAR)
print('Connected to IMUs')


# Video capture setup
print("Checking camera ...")
if NEW_CAM:
    # Look for devices. Returns as soon as it has found the first device.
    device = discover_one_device(max_search_duration_seconds=10)
    if device is None:
        print(LINE_UP, end=LINE_CLEAR)
        sys.exit("No device found.")

    print(LINE_UP, end=LINE_CLEAR)
    print(f"Connected to {device}")

    cam_message = 'Using New Camera \n'

else :
    cap = cv2.VideoCapture(cam_num)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ret, frame = cap.read()
    if not ret: # If camera is unavailable :
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        for connection in connections:
            connection.close()
        print(LINE_UP, end=LINE_CLEAR)
        print(LINE_UP, end=LINE_CLEAR)
        sys.exit('Camera disconnected.')
    cam_message = f'Camera Number : {cam_num} \n'

print(LINE_UP, end=LINE_CLEAR)
print('Connected to Camera')

try :
    input('\nProgramme Ready, Press Enter to Start')
    for i in range(2) :
        print(f'Starting in {2-i}s')
        sleep(1)
        print(LINE_UP, end=LINE_CLEAR)
except KeyboardInterrupt :
    sys.exit('\nProgramme Stopped\n')

# start read from keyboard input
threading.Thread(target=key_listener, daemon=True).start()

input("\nReady. Press Enter to start recording...\n")
print("Press SPACE to pause/resume.")

sequence_length = 10    # Size of samples default 10
sample_counter = 0
frames_counter = 0

Start_Time = time()
try : # try except is to ignore the keyboard interrupt error
    message = f'Programme running   ctrl + C to stop\n\nClean Folder : {CleanFolder} \n' + cam_message
    print('\033c'+message)
    while True:
        if is_paused:
            sleep(0.1)
            continue

        sample_counter += 1
        # We create a folder with a csv file in it( csv with rotation matrix)
        folder = f"{root_directory}/Sample_{sample_counter}"
        os.makedirs(folder)
        #csv_file = open(f"{folder}/imu.csv", 'w', newline='')
        with open(f"{folder}/imu.csv", 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            #based on that caculate new csv with joint angle
            #csv_file.close()

            for i in range(sequence_length):
                frames_counter += 1
                while time() - Start_Time < frames_counter / fps:
                    sleep(0.001)

                row = []
                for imu_id in IMU_IDS:
                    row += imu_data[imu_id]["gyr"] + imu_data[imu_id]["acc"] + sum(imu_data[imu_id]["rotm"], [])
                writer.writerow(row)

                if NEW_CAM :
                    # ret, frame = cap.read() 
                    bgr_pixels, frame_datetime = device.receive_scene_video_frame()
                    # ret = 
                    frame = bgr_pixels # TODO Possible source of error, check conversion
                else :
                    ret, frame = cap.read()
                    if not ret: # If camera is unavailable :
                        # Release resources
                        cap.release()
                        cv2.destroyAllWindows()
                        csv_file.close()
                        for connection in connections:
                            connection.close()
                        print('\nCamera disconnected')
                        raise KeyboardInterrupt

                if PRINT_IMU:
                    #print(f"Frame {frames_counter}, Sample {sample_counter}")
                    if frames_counter%window_size == 0 :
                        print('\033c'+message)

                # Add image
                cv2.imwrite(f"{folder}/frame_{frames_counter}.jpg", frame)

        #csv_file.close()
        process_imu_to_new(folder)

        # We delete the folders as we go so that we don't saturate
        if sample_counter > buffer:
            del_folder = f"{root_directory}/Sample_{sample_counter - buffer}"
            for file in os.listdir(del_folder):
                os.remove(os.path.join(del_folder, file))
            os.rmdir(del_folder)

except KeyboardInterrupt:
    t = round(time() - Start_Time, 4)
    print(f"\nStopped after {frames_counter} frames in {format_time(t)} — FPS: {frames_counter / t:.2f}")
    try:
        csv_file.close()
    except:
        pass
    if CleanFolder:
        for folders_left in os.listdir(root_directory) :
            for files_left in os.listdir(f"{root_directory}/{folders_left}"):
                os.remove(os.path.join(f'{root_directory}/{folders_left}', files_left))
            os.rmdir(f"{root_directory}/{folders_left}")
        os.rmdir(root_directory)
        if not NEW_CAM:
            cap.release()
        else:
            device.close()
        for c in connections:
            c.close()
        cv2.destroyAllWindows()


if __name__ == "__main__" :
    print('\nProgramme Stopped\n')