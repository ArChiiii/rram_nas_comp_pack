from .resnet.resnet import ResNet18
from .mobilenet.mobilenetv3_small import MobileNetV3_Small
from .squeeznet import SqueezeNet

model_dict = {
    "mbv3_small": MobileNetV3_Small(input_channels = 3, num_classes = 10),
    "squeezenet": SqueezeNet("1_1",10),
    "resnet18": ResNet18()
}

def get_model(model_str):
    return model_dict.get(model_str, None)
