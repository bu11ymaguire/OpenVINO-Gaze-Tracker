import cv2
import numpy as np

def draw_face_box(frame, face_box, color=(0, 255, 0)):
    x_min, y_min, x_max, y_max = face_box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

def draw_landmarks(frame, face_box, landmarks, color=(255, 255, 0), radius=2):
    x_min, y_min, x_max, y_max = face_box
    w_box = x_max - x_min
    h_box = y_max - y_min
    
    # Landmarks are normalized to [0,1] relative to the face box
    for i in range(0, len(landmarks), 2):
        lx = int(landmarks[i] * w_box + x_min)
        ly = int(landmarks[i+1] * h_box + y_min)
        cv2.circle(frame, (lx, ly), radius, color, -1)

def draw_axis(frame, head_pose, origin, scale=50):
    yaw, pitch, roll = head_pose
    x, y = origin
    
    # Convert angles from degrees to radians
    import math
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)
    
    # X-axis (Red) - Points to right
    x1 = scale * (math.cos(yaw_rad) * math.cos(roll_rad))
    y1 = scale * (math.cos(pitch_rad) * math.sin(roll_rad) + math.cos(roll_rad) * math.sin(pitch_rad) * math.sin(yaw_rad))
    cv2.line(frame, (x, y), (int(x + x1), int(y + y1)), (0, 0, 255), 3)
    
    # Y-axis (Green) - Points down
    x2 = scale * (-math.cos(yaw_rad) * math.sin(roll_rad))
    y2 = scale * (math.cos(pitch_rad) * math.cos(roll_rad) - math.sin(pitch_rad) * math.sin(yaw_rad) * math.sin(roll_rad))
    cv2.line(frame, (x, y), (int(x + x2), int(y + y2)), (0, 255, 0), 3)
    
    # Z-axis (Blue) - Points forward
    x3 = scale * (math.sin(yaw_rad))
    y3 = scale * (-math.cos(yaw_rad) * math.sin(pitch_rad))
    cv2.line(frame, (x, y), (int(x + x3), int(y + y3)), (255, 0, 0), 2)

def draw_gaze(frame, eye_pos, gaze_vector, length=100):
    x, y = eye_pos
    gx, gy, gz = gaze_vector
    # Gaze vector is usually normalized. Project to 2D image plane?
    # Simple orthographic projection for visualization:
    # x' = x + gx * length
    # y' = y - gy * length (Y is down in image, up in 3D usually, need to check coord sys)
    # OpenVINO sys: x-right, y-down, z-forward
    
    end_x = int(x + gx * length)
    end_y = int(y + gy * length)
    
    cv2.arrowedLine(frame, (x, y), (end_x, end_y), (255, 0, 255), 2)

def draw_results(frame, results):
    for res in results:
        box = res["box"]
        draw_face_box(frame, box)
        
        if "landmarks" in res:
            draw_landmarks(frame, box, res["landmarks"])
            
        if "head_pose" in res:
            center_x = (box[0] + box[2]) // 2
            center_y = (box[1] + box[3]) // 2
            draw_axis(frame, res["head_pose"], (center_x, center_y))
            
        if "gaze" in res and "eyes" in res:
            le_c, re_c = res["eyes"] # Relative to face box? No, logic in pipeline seemed relative to face img
            
            # Convert face-local coords to global coords
            fx, fy = box[0], box[1]
            le_global = (fx + le_c[0], fy + le_c[1])
            re_global = (fx + re_c[0], fy + re_c[1])
            
            draw_gaze(frame, le_global, res["gaze"])
            draw_gaze(frame, re_global, res["gaze"])

        # Draw Text (Age, Gender, Emotion)
        text_lines = []
        if "age" in res and "gender" in res:
            text_lines.append(f"{res['gender']}, {res['age']}")
        if "emotion" in res:
            text_lines.append(f"{res['emotion']}")
            
        if text_lines:
            x_min, y_min = box[0], box[1]
            for i, line in enumerate(text_lines):
                y_pos = y_min - 10 - (i * 20)
                if y_pos < 20: # If too close to top edge, draw inside/below
                    y_pos = y_min + 20 + (i * 20)
                cv2.putText(frame, line, (x_min, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
