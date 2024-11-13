import myo
from collections import deque
import numpy as np
from pynput import keyboard

class Listener(myo.DeviceListener):

    def __init__(self):
        self.connect = False
        self.device_name = None
        self.battery_level = None

    def on_connected(self, event):
        #print("Connected device: '{}'.".format(event.device_name))
        self.connect = True
        self.device_name = event.device_name
        event.device.vibrate(myo.VibrationType.short)
        event.device.request_battery_level()

    def on_battery_level(self, event):
        #print("Battery level:", event.battery_level, "%")
        self.battery_level = event.battery_level

    def on_pose(self, event):
        if event.pose == myo.Pose.double_tap:
            return False


class DataCollector(myo.DeviceListener):

    def __init__(self, n=40):

        self.emg_data_queue = deque(maxlen=n)
        self.emg_data_curr = [0,0,0,0,0,0,0,0]
        self.imu_data_curr = []
        
    def on_connected(self, event):
        # Request the signal strength from the device
        event.device.request_rssi()
        # Enable streaming the emg values
        event.device.stream_emg(True)

    def on_emg(self, event):
        self.emg_data_curr = event.emg
        

    def on_orientation(self, event):
        self.imu_data_curr = event.orientation

    def update_que(self):
        self.emg_data_queue.append(self.emg_data_curr)

    def clear_que(self):
        self.emg_data_queue.clear()


def initialize_myo(sdk_path):
    print("===========================")
    try:
        myo.init(sdk_path=sdk_path)
    except:
        print("Cannot find myo-sdk, wrong SDK path")

    hub, listener = None, None
    retries = 0
    
    try:
        hub = myo.Hub()
        listener = Listener()
    except:
        print("Unable to connect to Myo Connect.")
        print("Please run Myo Connect.exe")

    if hub != None:
        try:
            hub.run(listener.on_event, 500)
        finally:
            if listener.connect==False:
                print("Cannot find Myo devices.")
            else:
                print("Connected device: '{}'.".format(listener.device_name))
                print("Battery level: {}%".format(listener.battery_level))
                
    print("===========================")

    return hub, listener


# ==== keyboard input
def listen_keyboard(k_events):
    key_input = None
    break_loop = False

    # Block at most 0 second
    k_event = k_events.get(0)   # Get Press Event
    # _ = k_events.get(0)   # Get Release Event

    if k_event != None:
        if k_event.key == keyboard.Key.esc:
            print('Received event "{}". Stop keyboard listening'.format(k_event))
            # break
            break_loop = True
        else:
            if type(k_event)==keyboard.Events.Release:
                try:
                    key_input = k_event.key.char
                except AttributeError:
                    key_input = k_event.key
    
    return key_input, break_loop


def initialize_myo_tk(sdk_path):
    connect = False
    try:
        myo.init(sdk_path=sdk_path)
    except:
        txt_print = "Cannot find Myo-sdk.\nWrong SDK path"

    hub, listener = None, None
    retries = 0
    
    try:
        hub = myo.Hub()
        listener = Listener()
    except:
        txt_print = "Unable to connect to Myo Connect.\nPlease run Myo Connect.exe"

    if hub != None:
        try:
            hub.run(listener.on_event, 500)
        finally:
            if listener.connect==False:
                txt_print = "Cannot find Myo devices.\n "
            else:
                connect = True
                txt_print = "Connected device: '{}'\n".format(listener.device_name)
                txt_print += "Battery level: {}%".format(listener.battery_level)

    return hub, listener, txt_print, connect


def switch_mode(controller, id):
    if controller == 4:
        id -= 1
        if id == 1:
            id = 4
    elif controller == 5:
        id += 1
        if id==5:
            id = 2 
    return id