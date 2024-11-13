from dataclasses import dataclass
# from peaceful_pie.unity_comms import UnityComms
from unity_comms_my import UnityComms

# unity

@dataclass
class MyVector3:
    x: float
    y: float
    z: float

def connect_unity(port=9000):
    print("===========================")
    unity_joint = UnityComms(port=port)

    try:
        a = unity_joint.ConnectionTest()
    except:
        print("Cannot connect Unity. Wrong port or did not start to play in unity.")
    else:
        print("Unity connected.")
    print("===========================")


def connect_unity_tk(port=9000):
    connect = False
    get=0
    unity_joint = UnityComms(port=port)
    try:
        get = unity_joint.ConnectionTest()
    finally:
        if get == 1:
            txt_print = "Unity connected.\n"
            connect = True
        else:
            txt_print = "Cannot connect Unity.\nWrong port or not playing in Unity."
            if connect==False:
                unity_joint = None
    
    return unity_joint, txt_print, connect

def check_controller_tk(unity_joint):
    connect = False
    try:
        is_found = unity_joint.CheckVRDevices()
        print(type(is_found['Item1']), is_found['Item1'])
        print(type(is_found['Item2']), is_found['Item2'])
        controller = unity_joint.ReturnInput()
    finally:
        if is_found['Item2']:
            txt_print = is_found['Item1'] + "\n"
            connect = is_found['Item2']
        else:
            txt_print = is_found['Item1'] + "\nPlease check the connection in SteamVR."
    
    return txt_print, connect