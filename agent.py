from ultralytics import YOLO
import ultralytics
import time
import supervision as sv
import cv2
import math
import numpy as np

class Agent:
    def __init__(self):
        ultralytics.checks()
        self.model = YOLO(r'models/best.pt')
        #self.model = YOLO(r'resources/yolo11n-obb.pt')

    def train(self):
        return self.model.train(data='resources/datasets/rally/data.yaml', epochs=100, imgsz=640)

    def _angle_normalize(self, angle, relative=0):
        angle_norm = angle - relative
        if angle_norm > 180:
            angle_norm -= 360
        elif angle_norm < -180:
            angle_norm += 360

        return angle_norm

    def process_vehicle_data(self, data):
        x1 = data[0][0]
        y1 = data[0][1]
        x2 = data[1][0]
        y2 = data[1][1]
        
        x3 = data[2][0]
        x4 = data[3][0]
        y3 = data[2][1]
        y4 = data[3][1]
        
        min_y = min(y1, y2, y3, y4)
        max_y = max(y1, y2, y3, y4)

        min_x = min(x1, x2, x3, x4)
        max_x = max(x1, x2, x3, x4)

        dx = x2 - x1
        dy = y2 - y1
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        angle_normalized = self._angle_normalize(angle_deg, relative=90)

        return min_x, max_x, min_y, max_y, angle_normalized
    
    def _get_longest_contour(self, contours, gap_treshold=20):
        if not contours:
            return None

        contour_ranges = []
        for cnt in contours:
            x = cnt[:, 0, 0]
            contour_ranges.append((x.min(), x.max(), cnt))

        sorted_contours = sorted(contour_ranges, key=lambda x: x[0])

        merged_groups = []
        current_group = list(sorted_contours[0])

        for (min_x, max_x, cnt) in sorted_contours[1:]:
            if min_x - current_group[1] <= gap_treshold:
                # Merge if matches the treshold
                current_group[1] = max(current_group[1], max_x)
                current_group[2] = np.vstack((current_group[2], cnt))
            else:
                # Start a new group
                merged_groups.append(current_group)
                current_group = [min_x, max_x, cnt]

        merged_groups.append(current_group)

        # Find the group with max horizontal span
        longest_group = max(merged_groups, key=lambda g: g[1] - g[0])

        return longest_group[2] # Return combined contour points

    def detect_rally(self, frame):
        result = self.model(frame)
        detections = sv.Detections.from_ultralytics(result[0])
        #obb_annotator = sv.OrientedBoxAnnotator()
        #annotated_frame = obb_annotator.annotate(scene=frame, detections=detections)
        if len(detections) < 1:
            return 0
        
        #angle, min_y = self.get_angle(detections.data['xyxyxyxy'][0])
        #sv.plot_image(image=annotated_frame, size=(16,16))
        return detections

    def _check_ground_presence(self, ground_contour, box):
        points = ground_contour[:, 0, :]
        in_x = (points[:, 0] >= box[0, 0]) & (points[:, 0] <= box[1, 0])
        in_y = (points[:, 1] >= box[0, 1]) & (points[:, 1] <= box[3, 1])
                        
        return np.any(in_x & in_y)
    
    def process_ground_data(self, ground_contour, start_x, end_x):
        points = ground_contour[:, 0, :]

        # Find points within the search area (start_x, end_x)
        mask = (points[:, 0] > start_x) & (points[:, 0] < end_x)
        relevant_points = points[mask]

        if len(relevant_points) < 2:
            return None

        left_point = relevant_points[np.argmin(relevant_points[:, 0])]
        right_point = relevant_points[np.argmax(relevant_points[:, 0])]

        dx = right_point[0] - left_point[0]
        dy = right_point[1] - left_point[1]
        angle = np.degrees(np.arctan2(-dy, dx))  # Negative dy because y increases downward
        
        return angle

    def detect_ground(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = np.float32(frame)
        #frame = cv2.cornerHarris(frame, 9, 11, 0.04)
        
        #ret, tresh = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)
        blur = cv2.GaussianBlur(gray, (1,1), cv2.BORDER_DEFAULT)
        canny = cv2.Canny(blur,100,200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        closed_edges = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contour = max(contours, key=lambda c: max([pt[0][0] for pt in c]) - min([pt[0][0] for pt in c]))
        
        height = frame.shape[0]
        ground_contours = [cnt for cnt in contours if np.max(cnt[:, 0, 1]) > height * 0.40]
        ground_contour = max(ground_contours, key=lambda c: np.ptp(c[:, 0, 0]))  # ptp = max - min
        #print(contours)
        #contour = self._get_longest_contour(contours, 1)       
        # Optional: Get the actual span value
        #x_coords = [pt[0][0] for pt in contour]
        #span = max(x_coords) - min(x_coords)
        #print(f"Horizontal span: {span} pixels")

        #frame = cv2.drawContours(frame, [ground_contour],-1 , (0,255,0), 2)

        return ground_contour

    def process_frame(self, frame):
        ground_contour = self.detect_ground(frame)
        rally_contour = self.detect_rally(frame)
        if not isinstance(rally_contour, sv.detection.core.Detections) or not isinstance(ground_contour, np.ndarray):
           print('Could not find contours / Corrupt data', type(rally_contour), type(ground_contour))
           return None, None, None

        rally_min_x, rally_max_x, rally_min_y, rally_max_y, rally_angle = self.process_vehicle_data(rally_contour.data['xyxyxyxy'][0])
        if abs(rally_angle) > 160:
            rally_angle = self._angle_normalize(rally_angle, relative=180) # Unflips on cv mistake
        try:
             ground_angle = -1 * self.process_ground_data(ground_contour, start_x=rally_min_x, end_x=rally_max_x)
             if ground_angle == None:
                 print('Corrupt ground data, continuing...')
                 return None, None, None
        except:
            return None, None, None
        bbox = np.array([[rally_min_x, rally_max_y],    # Top-left
                         [rally_max_x, rally_max_y],    # Top-right
                         [rally_max_x, rally_min_y],    # Bottom-right
                         [rally_min_x, rally_min_y]])   # Bottom-left

        rally_on_ground = self._check_ground_presence(ground_contour, bbox)
        rally_relative_angle = self._angle_normalize(angle=rally_angle, relative=ground_angle)
        print(rally_angle, rally_relative_angle, rally_on_ground, ground_angle)
        return rally_angle, rally_relative_angle, rally_on_ground

