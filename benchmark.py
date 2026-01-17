
import cv2
import time
import numpy as np
import os
from src.model_loader import ModelManager
from src.pipeline import GazePipeline

def benchmark_device(device_name, num_frames=100):
    print(f"\n--- Benchmarking {device_name} ---")
    
    BASE_MODEL_DIR = os.path.join(os.getcwd(), "models", "intel")
    PRECISION = "FP16" # FP16 is optimal for GPU/NPU, usually fine for CPU too
    
    # Paths
    FACE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"face-detection-adas-0001/{PRECISION}/face-detection-adas-0001.xml")
    LANDMARK_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"facial-landmarks-35-adas-0002/{PRECISION}/facial-landmarks-35-adas-0002.xml")
    HEAD_POSE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"head-pose-estimation-adas-0001/{PRECISION}/head-pose-estimation-adas-0001.xml")
    GAZE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"gaze-estimation-adas-0002/{PRECISION}/gaze-estimation-adas-0002.xml")
    AGE_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"age-gender-recognition-retail-0013/{PRECISION}/age-gender-recognition-retail-0013.xml")
    EMOTION_MODEL_PATH = os.path.join(BASE_MODEL_DIR, f"emotions-recognition-retail-0003/{PRECISION}/emotions-recognition-retail-0003.xml")

    try:
        # Load Models
        start_load = time.time()
        model_mgr = ModelManager(device=device_name)
        model_mgr.load_model("face_detection", FACE_MODEL_PATH)
        model_mgr.load_model("landmarks", LANDMARK_MODEL_PATH)
        model_mgr.load_model("head_pose", HEAD_POSE_MODEL_PATH)
        model_mgr.load_model("gaze", GAZE_MODEL_PATH)
        model_mgr.load_model("age_gender", AGE_MODEL_PATH)
        model_mgr.load_model("emotion", EMOTION_MODEL_PATH)
        print(f"Model Loading Time: {time.time() - start_load:.2f}s")
        
        pipeline = GazePipeline(model_mgr)
        
        # Warmup
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Draw a fake face so pipeline runs all models (otherwise it skips if no face)
        # Actually pipeline skips if no face detected.
        # We need a real image or mock detection.
        # Let's mock detect_faces in pipeline or use an image with a face?
        # Using a black image will result in 0 faces -> FPS will be just Face Detection speed.
        # To test FULL pipeline, we need a face.
        
        # Solution: Capture one frame from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Cannot get webcam frame for benchmark. Using blank image (might skip downstream models).")
            frame = dummy_frame

        print("Warming up...")
        for _ in range(10):
            pipeline.run(frame)
            
        print(f"Running {num_frames} inference loops...")
        start_time = time.time()
        for _ in range(num_frames):
            pipeline.run(frame)
        end_time = time.time()
        
        avg_fps = num_frames / (end_time - start_time)
        print(f"Result: {avg_fps:.2f} FPS")
        return avg_fps

    except Exception as e:
        print(f"Benchmark Failed: {e}")
        return 0

if __name__ == "__main__":
    devices = ["CPU", "GPU"]
    # Check NPU if available (NPU compiling might take longer)
    try:
        from openvino.runtime import Core
        if "NPU" in Core().available_devices:
            devices.append("NPU")
    except:
        pass

    results = {}
    for device in devices:
        fps = benchmark_device(device)
        results[device] = fps
        
    print("\n=== Final Benchmark Results ===")
    best_device = max(results, key=results.get)
    for dev, fps in results.items():
        print(f"{dev}: {fps:.2f} FPS")
    print(f"Recommendation: Use {best_device}")
