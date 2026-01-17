import os

def record_model_info():
    # Configuration matches main.py
    PRECISION = "FP16"
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "intel")
    OUTPUT_FILE = os.path.join(SCRIPT_DIR, "first.txt")

    # Define the models used in the project
    models = [
        "face-detection-adas-0001",
        "facial-landmarks-35-adas-0002",
        "head-pose-estimation-adas-0001",
        "gaze-estimation-adas-0002",
        "age-gender-recognition-retail-0013",
        "emotions-recognition-retail-0003"
    ]

    print(f"Recording model information to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== OpenVINO Gaze Estimation Project Model List ===\n")
        f.write(f"Precision Used: {PRECISION}\n\n")
        
        for model_name in models:
            # Construct path to verify existence
            model_xml = os.path.join(BASE_MODEL_DIR, model_name, PRECISION, f"{model_name}.xml")
            
            if os.path.exists(model_xml):
                f.write(f"- Model Name: {model_name}\n")
                f.write(f"  - Precision: {PRECISION}\n")
                f.write(f"  - Path: {model_xml}\n")
                
                # Add brief description based on known info
                if "face-detection" in model_name:
                    desc = "Face Detection (MobileNet + SSD)"
                elif "facial-landmarks" in model_name:
                    desc = "Facial Landmarks (Regression CNN, 35 points)"
                elif "head-pose" in model_name:
                    desc = "Head Pose Estimation (Yaw, Pitch, Roll)"
                elif "gaze-estimation" in model_name:
                    desc = "Gaze Estimation (Multi-stream CNN)"
                elif "age-gender" in model_name:
                    desc = "Age & Gender Recognition (Multi-task CNN)"
                elif "emotions" in model_name:
                    desc = "Emotion Recognition (Classification)"
                else:
                    desc = "Unknown"
                
                f.write(f"  - Description: {desc}\n\n")
            else:
                f.write(f"- Model Name: {model_name} (NOT FOUND)\n\n")

    print("Done!")

if __name__ == "__main__":
    record_model_info()
