from django.apps import AppConfig
from tensorflow.keras.models import load_model
import os


class AutomatedgraderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'automatedgrader'
    model_path = model_path = os.path.join(os.path.dirname(__file__), 'ml_model', 'model_set1.h5')
    ml_model=None
    def ready(self):
        self.ml_model = load_model(self.model_path)

