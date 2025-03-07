from connection import Connection
import cv2
import numpy as np
from dataclasses import dataclass
from time import sleep
import agent-debug

@dataclass
class VehicleData:
    rotation: float
    state: str
    position: tuple[int, int]

class HCR2Helper:
    def __init__(self):
        self.connection = Connection()
        self.agent = agent.AgentDebug()

    def run(self):
        self.connection.vehicle_gas()
        prev_brake = None
        brake = False
        angle = 0
        prev_angle = 0
        while True:
            if abs(angle - prev_angle) < 45 and abs(relative_angle - angle) < 90:
                 if abs(angle) > 50:
                     brake=True
                 else:
                     brake=False
            else:
                brake=False
            
            self.connection.vehicle_brake(stop=brake)
            
            prev_brake = brake
            prev_angle = angle
            # Add control logic here based on data
            sleep(0.001)

    def debug(self):
        frame = cv2.imread(r'captures/20.png')
        #angle = self.agent.detect_rally(frame)
        angle, relative_angle, on_ground = self.agent.process_frame(frame)
        print(angle, relative_angle, on_ground)


if __name__ == "__main__":
    helper = HCR2Helper()
    helper.run()
