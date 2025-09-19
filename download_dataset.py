!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="2sWq8uAPuT7C8mm5gggs")
project = rf.workspace("crab-noqmx").project("crabe_bleu_detection")
version = project.version(6)
dataset = version.download("yolov8")
                
