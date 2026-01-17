import cv2
import numpy as np

class GazePipeline:
    def __init__(self, model_manager):
        self.mm = model_manager
        self.face_model = model_manager.get_model("face_detection")
        self.landmark_model = model_manager.get_model("landmarks")
        self.head_pose_model = model_manager.get_model("head_pose")
        self.gaze_model = model_manager.get_model("gaze")
        self.age_gender_model = model_manager.get_model("age_gender")
        self.emotion_model = model_manager.get_model("emotion")
        
        # Input shapes
        self.n_fd, self.c_fd, self.h_fd, self.w_fd = self.face_model.inputs[0].shape
        self.h_lm, self.w_lm = 60, 60 
        self.h_hp, self.w_hp = 60, 60
        self.h_gz, self.w_gz = 60, 60
        self.h_ag, self.w_ag = 62, 62
        self.h_em, self.w_em = 64, 64

    def preprocess(self, frame, h, w):
        """Prepares the frame for network input."""
        resized = cv2.resize(frame, (w, h))
        transposed = resized.transpose(2, 0, 1)
        input_data = np.expand_dims(transposed, axis=0)
        return input_data

    def detect_faces(self, frame, conf_threshold=0.5):
        """
        Runs face detection. Returns list of [x_min, y_min, x_max, y_max].
        """
        input_data = self.preprocess(frame, self.h_fd, self.w_fd)
        results = self.face_model.infer_new_request({0: input_data})
        
        detections = next(iter(results.values()))
        detections = detections.reshape(-1, 7)
        
        boxes = []
        h, w = frame.shape[:2]
        
        for detection in detections:
            conf = detection[2]
            if conf > conf_threshold:
                x_min = int(detection[3] * w)
                y_min = int(detection[4] * h)
                x_max = int(detection[5] * w)
                y_max = int(detection[6] * h)
                
                # Clip
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                
                boxes.append([x_min, y_min, x_max, y_max])
        return boxes

    def get_landmarks(self, face_img):
        """
        Returns 35 landmarks (flattened array of 70 floats).
        """
        input_data = self.preprocess(face_img, self.h_lm, self.w_lm)
        results = self.landmark_model.infer_new_request({0: input_data})
        return next(iter(results.values())).flatten()

    def get_head_pose(self, face_img):
        """
        Returns (yaw, pitch, roll) angles.
        """
        input_data = self.preprocess(face_img, self.h_hp, self.w_hp)
        results = self.head_pose_model.infer_new_request({0: input_data})
        
        yaw = results[self.head_pose_model.outputs[0]][0][0]
        pitch = results[self.head_pose_model.outputs[1]][0][0]
        roll = results[self.head_pose_model.outputs[2]][0][0]
        return yaw, pitch, roll
    
    def get_age_gender(self, face_img):
        """
        Returns (age, gender_str).
        Gender: 0 - Female, 1 - Male
        """
        input_data = self.preprocess(face_img, self.h_ag, self.w_ag)
        results = self.age_gender_model.infer_new_request({0: input_data})
        
        # Outputs: 'age_conv3', 'prob'
        # Determine keys dynamically or rely on order if known.
        # Often: 
        #   age output shape [1, 1, 1, 1] -> Multiply by 100
        #   gender output shape [1, 2, 1, 1]
        
        age = 0
        gender_prob = [0.5, 0.5]
        
        for node, res in results.items():
            if res.shape == (1, 1, 1, 1):
                age = res[0,0,0,0] * 100
            elif res.shape == (1, 2, 1, 1):
                gender_prob = res[0,:,0,0]
        
        gender = "Female" if gender_prob[0] > gender_prob[1] else "Male"
        return int(age), gender

    def get_emotion(self, face_img):
        """
        Returns emotion string.
        Classes: neutral, happy, sad, surprise, anger
        """
        input_data = self.preprocess(face_img, self.h_em, self.w_em)
        results = self.emotion_model.infer_new_request({0: input_data})
        
        # Output: [1, 5, 1, 1]
        probs = next(iter(results.values())).flatten()
        classes = ["neutral", "happy", "sad", "surprise", "anger"]
        return classes[np.argmax(probs)]

    def crop_eye(self, face_img, eye_center_norm, scale=2.5):
        """
        Crops eye based on normalized center in face image.
        Uses a fixed and constant size box around eye.
        """
        fh, fw = face_img.shape[:2]
        cx = int(eye_center_norm[0] * fw)
        cy = int(eye_center_norm[1] * fh)
        
        crop_size = int(fw * 0.4) 
        half_size = crop_size // 2
        
        x1 = max(0, cx - half_size)
        y1 = max(0, cy - half_size)
        x2 = min(fw, cx + half_size)
        y2 = min(fh, cy + half_size)
        
        eye_img = face_img[y1:y2, x1:x2]
        return eye_img, (cx, cy)

    def get_gaze(self, left_eye_img, right_eye_img, head_pose_angles):
        """
        Returns gaze vector (x, y, z).
        """
        le_input = self.preprocess(left_eye_img, self.h_gz, self.w_gz)
        re_input = self.preprocess(right_eye_img, self.h_gz, self.w_gz)
        hp_input = np.array(head_pose_angles).reshape(1, 3)
        
        inputs = {
            "left_eye_image": le_input,
            "right_eye_image": re_input,
            "head_pose_angles": hp_input
        }
        
        results = self.gaze_model.infer_new_request(inputs)
        gaze_vector = next(iter(results.values()))[0]
        return gaze_vector

    def run(self, frame):
        """
        Main pipeline: Face -> [Landmarks, Head Pose, Age/Gender, Emotion] -> Gaze
        """
        faces = self.detect_faces(frame)
        
        results = []
        for face_box in faces:
            # 1. Crop face
            x_min, y_min, x_max, y_max = face_box
            if x_max - x_min == 0 or y_max - y_min == 0: continue
            
            face_img = frame[y_min:y_max, x_min:x_max]
            
            # 2. Parallel Inferences
            lm = self.get_landmarks(face_img)
            yaw, pitch, roll = self.get_head_pose(face_img)
            age, gender = self.get_age_gender(face_img)
            emotion = self.get_emotion(face_img)
            
            # 3. Gaze Estimation
            le_center = ((lm[0] + lm[2])/2, (lm[1] + lm[3])/2)
            re_center = ((lm[4] + lm[6])/2, (lm[5] + lm[7])/2)
            
            left_eye_img, le_c_px = self.crop_eye(face_img, le_center)
            right_eye_img, re_c_px = self.crop_eye(face_img, re_center)
            
            gaze_vector = None
            if left_eye_img.size > 0 and right_eye_img.size > 0:
                gaze_vector = self.get_gaze(left_eye_img, right_eye_img, (yaw, pitch, roll))
            
            results.append({
                "box": face_box,
                "landmarks": lm,
                "head_pose": (yaw, pitch, roll),
                "gaze": gaze_vector,
                "eyes": (le_c_px, re_c_px),
                "age": age,
                "gender": gender,
                "emotion": emotion
            })
            
        return results
