from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .myresnet import myresnet8, myresnet14, myresnet20, myresnet32, myresnet44, myresnet56, myresnet110, myresnet8x4, myresnet32x4
from .resnetv2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .myresnetv2 import myResNet18, myResNet34, myResNet50, myResNet101, myResNet152
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2, wrn_10_2
from .mywrn import mywrn_16_1, mywrn_16_2, mywrn_40_1, mywrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .densenet import densenet100, densenet40
from .resnext import resnext101, resnext29
from .myresnext import myresnext101, myresnext29

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'wrn_10_2': wrn_10_2,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,

    'myresnet8': myresnet8,
    'myresnet14': myresnet14,
    'myresnet20': myresnet20,
    'myresnet32': myresnet32,
    'myresnet44': myresnet44,
    'myresnet56': myresnet56,
    'myresnet110': myresnet110,
    'myresnet8x4': myresnet8x4,
    'myresnet32x4': myresnet32x4,

    'mywrn_16_1': mywrn_16_1,
    'mywrn_16_2': mywrn_16_2,
    'mywrn_40_1': mywrn_40_1,
    'mywrn_40_2': mywrn_40_2,

    'myResNet18': myResNet18,
    'myResNet34': myResNet34,
    'myResNet50': myResNet50,
    'myResNet101': myResNet101,
    'myResNet152': myResNet152,

    'densenet100': densenet100,
    'densenet40': densenet40,

    'resnext101': resnext101,
    'resnext29': resnext29,
    'myresnext101': myresnext101,
    'myresnext29': myresnext29,
}
