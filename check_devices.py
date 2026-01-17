from openvino.runtime import Core

def check_devices():
    core = Core()
    devices = core.available_devices
    print(f"Available Devices: {devices}")
    
    for device in devices:
        try:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"  {device}: {device_name}")
        except:
            print(f"  {device}: Unknown Name")

if __name__ == "__main__":
    check_devices()
