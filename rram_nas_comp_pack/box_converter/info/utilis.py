
from .layer_info import LayerInfo
from rram_nas_comp_pack.box_converter.info.LayerBox import LayerBox

def Conv2dBoxInfo(layer_info: LayerInfo, exc_order:int, weight_precision):
    output_size = layer_info.output_size
    input_size = layer_info.input_size
    kernel_size = layer_info.kernel_size

    groups = layer_info.module.groups
    if groups != 1: # depthwise layer
        # 1 cycle 1 channel process
        h = kernel_size[0]*kernel_size[1]
        w = output_size[1]
        cycle = output_size[1] * output_size[2] * output_size[3]
        return LayerBox(h, w, cycle, exc_order, weight_precision)

    #same for normal and pointwise 
    h = kernel_size[0]*kernel_size[1]*input_size[1]
    w = output_size[1]
    cycle = output_size[2] * output_size[3]
    return LayerBox(h, w, cycle, exc_order, weight_precision)


def LinearBoxInfo(layer_info: LayerInfo, exc_order:int, weight_precision):
    output_size = layer_info.output_size
    input_size = layer_info.input_size

    h = input_size[1]
    w = output_size[1]
    cycle = 1
    return LayerBox(h, w, cycle, exc_order, weight_precision)