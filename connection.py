import scrcpy
from ppadb.client import Client as AdbClient
import time

class Connection:
    def __init__(self):
        self.device = self.__connect()
        self.capdir = r'C:\Users\Dominik\Documents\Python Projects\HCR2_Helper\resources\temp'
        self.client = scrcpy.Client(device=self.device.serial)
        self.client.start(threaded=True)

    def __connect(self):
        client = AdbClient(host="127.0.0.1", port=5037)
        devices = client.devices()

        if devices:
            return devices[0]
        return None

    def capture(self):
        # A slower method -
        # self.device.shell(f'adb exec-out screencap -p >{self.capdir}\cap.png')
        #sbytes = self.device.screencap()
        #return Image.open(io.BytesIO(sbytes))
        return self.client.last_frame

    def devtap(self, action, x, y, duration: int):
        """
        Inputs data to device as screen-taps
        """
        self.client.control.touch(x, y, action=action)
        print(x,y,action, duration*0.001)
        time.sleep(duration*0.001)
    
    def vehicle_gas(self):
        self.client.control.touch(2000,900, action=scrcpy.ACTION_DOWN, touch_id=0)
    
    def vehicle_brake(self, stop):
        if stop == False:
            self.client.control.touch(200, 900, action=scrcpy.ACTION_UP, touch_id=1)
        else:
            self.client.control.touch(200, 900, action=scrcpy.ACTION_DOWN, touch_id=1)
            time.sleep(0.01)
