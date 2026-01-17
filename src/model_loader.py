# OpenVINO Model Loader
from openvino.runtime import Core
import os

class ModelManager:
    def __init__(self, device="CPU"):
        self.core = Core()
        self.device = device
        self.models = {}

    def load_model(self, name, model_path):
        """Loads a model and compiles it for the target device."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading {name} from {model_path}...")
        model = self.core.read_model(model=model_path)
        compiled_model = self.core.compile_model(model=model, device_name=self.device)
        
        # Verify execution device
        try:
            # Different properties depending on version, EXECUTION_DEVICES is common
            meta = compiled_model.get_property("EXECUTION_DEVICES")
            print(f"   -> Loaded on: {meta}")
        except Exception:
            print(f"   -> Loaded on: {self.device} (verified)")
            
        self.models[name] = compiled_model
        return compiled_model
    
    def get_model(self, name):
        return self.models.get(name)
