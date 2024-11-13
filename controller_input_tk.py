import openvr
import time



# controller input

def get_controller_ids(vrsys=None):
    if vrsys is None:
        vrsys = openvr.VRSystem()
    else:
        vrsys = vrsys
    left, right = None, None
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        device_class = vrsys.getTrackedDeviceClass(i)
        if device_class == openvr.TrackedDeviceClass_Controller:
            role = vrsys.getControllerRoleForTrackedDeviceIndex(i)
            if role == openvr.TrackedControllerRole_RightHand:
                right = i
            if role == openvr.TrackedControllerRole_LeftHand:
                left = i
    return left, right


def from_controller_state_to_dict(pControllerState):
    # docs: https://github.com/ValveSoftware/openvr/wiki/IVRSystem::GetControllerState
    d = {}
    d['unPacketNum'] = pControllerState.unPacketNum
    # on trigger .y is always 0.0 says the docs
    d['trigger'] = pControllerState.rAxis[1].x
    # 0.0 on trigger is fully released
    # -1.0 to 1.0 on joystick and trackpads
    d['trackpad_x'] = pControllerState.rAxis[0].x
    d['trackpad_y'] = pControllerState.rAxis[0].y
    # These are published and always 0.0
    # for i in range(2, 5):
    #     d['unknowns_' + str(i) + '_x'] = pControllerState.rAxis[i].x
    #     d['unknowns_' + str(i) + '_y'] = pControllerState.rAxis[i].y
    d['ulButtonPressed'] = pControllerState.ulButtonPressed
    d['ulButtonTouched'] = pControllerState.ulButtonTouched
    # To make easier to understand what is going on
    # Second bit marks menu button
    d['menu_button'] = bool(pControllerState.ulButtonPressed >> 1 & 1)
    # 32 bit marks trackpad
    d['trackpad_pressed'] = bool(pControllerState.ulButtonPressed >> 32 & 1)
    d['trackpad_touched'] = bool(pControllerState.ulButtonTouched >> 32 & 1)
    # third bit marks grip button
    d['grip_button'] = bool(pControllerState.ulButtonPressed >> 2 & 1)
    # System button can't be read, if you press it
    # the controllers stop reporting
    return d


def detect_edge(pre_sig, curr_sig):
    if pre_sig==False and curr_sig==True:
        return 1
    else:
        return 0


def detect_edge2(pre_sig, curr_sig):
    if pre_sig < 1 and curr_sig == 1:
        return 1
    else:
        return 0


def initialize_openvr(max_init_retries=4):
    retries = 0
    print("===========================")
    print("Initializing OpenVR...")
    while retries < max_init_retries:
        try:
            openvr.init(openvr.VRApplication_Scene)
            break
        except openvr.OpenVRError as e:
            print("Error when initializing OpenVR (try {} / {})".format(
                    retries + 1, max_init_retries))
            print(e)
            retries += 1
            time.sleep(2.0)
    else:
        print("Could not initialize OpenVR, aborting.")
        print("Make sure the system is correctly plugged, you can also try")
        print("to do:")
        print("killall -9 vrcompositor vrmonitor vrdashboard")
        print("Before running this program again.")
        exit(0)

    print("Success!")
    print("===========================")
    vrsystem = openvr.VRSystem()
    return vrsystem


def detect_controllers(vrsystem):
    # 检测到一个就行
    left_id, right_id = None, None
    print("===========================")
    print("Waiting for controllers...")
    try:
        while left_id is None or right_id is None:
            left_id, right_id = get_controller_ids(vrsystem)
            if left_id or right_id:
                break
            print("Waiting for controllers...")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Control+C pressed, shutting down...")
        openvr.shutdown()

    print("Left controller ID: " + str(left_id))
    print("Right controller ID: " + str(right_id))
    print("===========================")
    return left_id, right_id


def detect_input(vrsystem, d_pre, controller_id):

    if controller_id != None:
        _, pControllerState = vrsystem.getControllerState(controller_id)
        d = from_controller_state_to_dict(pControllerState)
        d_curr = d['trackpad_pressed']

        val_input = detect_edge(d_pre, d_curr)
        d_pre = d_curr
        
    else:
        val_input = 0
    
    return d_pre, val_input


def initialize_openvr_tk(max_init_retries=1):
    retries = 0
    vrsystem = None
    connect = False
    while retries < max_init_retries:
        try:
            openvr.init(openvr.VRApplication_Scene)
            vrsystem = openvr.VRSystem()
            txt_print = "VR controller connected."
            connect = True
            break
        except openvr.OpenVRError as e:
            # print("Error when initializing OpenVR (try {} / {})".format(
            #         retries + 1, max_init_retries))
            # print(e)
            retries += 1
            time.sleep(1.0)

    else:
        txt_print = "Could not initialize OpenVR, aborting.\nMake sure the system is correctly plugged."
    return vrsystem, txt_print, connect


