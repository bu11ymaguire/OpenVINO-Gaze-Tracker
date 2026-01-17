import cv2
import sys
import os
from src.model_loader import ModelManager
from src.pipeline import GazePipeline
from src.utils import draw_results

def main():
    # Configuration
    # Use FP16 for GPU/NPU, FP32 for CPU usually, but modern CPUs handle FP16 well via conversion or AVX512_BF16
    # We downloaded everything.
    PRECISION = "FP16" 
    DEVICE = "GPU" # Optimized based on benchmark results (Avg ~140 FPS)
    
    # Use paths relative to this script file, not the current working directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "intel")
    
    # Paths to models
    FACE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"face-detection-adas-0001/{PRECISION}/face-detection-adas-0001.xml")
    LANDMARK_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"facial-landmarks-35-adas-0002/{PRECISION}/facial-landmarks-35-adas-0002.xml")
    HEAD_POSE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"head-pose-estimation-adas-0001/{PRECISION}/head-pose-estimation-adas-0001.xml")
    GAZE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"gaze-estimation-adas-0002/{PRECISION}/gaze-estimation-adas-0002.xml")
    AGE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"age-gender-recognition-retail-0013/{PRECISION}/age-gender-recognition-retail-0013.xml")
    EMOTION_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"emotions-recognition-retail-0003/{PRECISION}/emotions-recognition-retail-0003.xml")
    
    # Initialize
    try:
        print(f"Initializing OpenVINO on {DEVICE}...")
        model_mgr = ModelManager(device=DEVICE)
        
        # Load Models
        model_mgr.load_model("face_detection", FACE_MODEL_PATH)
        model_mgr.load_model("landmarks", LANDMARK_MODEL_PATH)
        model_mgr.load_model("head_pose", HEAD_POSE_MODEL_PATH)
        model_mgr.load_model("gaze", GAZE_MODEL_PATH)
        model_mgr.load_model("age_gender", AGE_MODEL_PATH)
        model_mgr.load_model("emotion", EMOTION_MODEL_PATH)
        
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize Pipeline
    pipeline = GazePipeline(model_mgr)
    
    # FPS Calculation
    import time
    prev_time = 0
    
    print("Starting Loop. Press ESC to exit.")
    while True:
        curr_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = pipeline.run(frame)

        # Visualization
        draw_results(frame, results)
        
        # Calculate and Draw FPS
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f} ({DEVICE})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Gaze Estimation Project', frame)

        if cv2.waitKey(1) == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
