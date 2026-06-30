import os
from ultralytics import YOLO

class ModelManager:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.loaded_models = {}
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def get_available_models(self):
        """Scan and return available model files"""
        model_files = []
        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith('.pt'):
                    model_files.append(file)
        return model_files
    
    def load_model(self, model_name):
        """Load model from file"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found in {self.models_dir}")
        
        self.loaded_models[model_name] = YOLO(model_path)
        return self.loaded_models[model_name]