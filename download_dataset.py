# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("asmrgaming/deeplearning-ensemble-for-automated-acne-detection")

# print("Path to dataset files:", path)



from roboflow import Roboflow

rf = Roboflow(api_key="s6Fc74WomMHcaovf6rhY")
project = rf.workspace("kritsakorn").project("acne-kbm0q")
version = project.version(21)
dataset = version.download("yolov11")
                