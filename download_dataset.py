from roboflow import Roboflow
rf = Roboflow(api_key="s6Fc74WomMHcaovf6rhY")
project = rf.workspace("zest-iot").project("acne_dataset")
version = project.version(1)
dataset = version.download("yolov11")
                