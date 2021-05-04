"""
MIT License

Copyright (c) 2020 Mahdi S. Hosseini and Mathieu Tuli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import sys
mod_name = vars(sys.modules[__name__])['__name__']

if 'adas.' in mod_name:
    from .alexnet import alexnet as AlexNet
    from .densenet import densenet201 as DenseNet201, \
        densenet169 as DenseNet169, densenet161 as DenseNet161,\
        densenet121 as DenseNet121
    from .googlenet import googlenet as GoogLeNet
    from .inception import inception_v3 as InceptionV3
    from .mnasnet import mnasnet0_5 as MNASNet_0_5,\
        mnasnet0_75 as MNASNet_0_75, mnasnet1_0 as MNASNet_1,\
        mnasnet1_3 as MNASNet_1_3
    from .mobilenet import mobilenet_v2 as MobileNetV2
    from .mobilenet_cifar import MobileNetV2 as MobileNetV2CIFAR
    from .senet import SENet18 as SENet18CIFAR
    from .shufflenetv2_cifar import ShuffleNetV2 as ShuffleNetV2CIFAR
    from .resnet import resnet18 as ResNet18, resnet34 as ResNet34, \
        resnet50 as ResNet50, resnet101 as ResNet101, resnet152 as ResNet152, \
        resnext50_32x4d as ResNeXt50, resnext101_32x8d as ResNeXt101, \
        wide_resnet50_2 as WideResNet50, wide_resnet101_2 as WideResNet101
    from .resnet_cifar import ResNet34 as ResNet34CIFAR,\
        ResNet18 as ResNet18CIFAR, ResNet101 as ResNet101CIFAR, ResNet50 as ResNet50CIFAR
    from .resnext_cifar import ResNeXt29_1x64d as ResNeXtCIFAR
    from .shufflenetv2 import shufflenet_v2_x0_5 as ShuffleNetV2_0_5, \
        shufflenet_v2_x1_0 as ShuffleNetV2_1, \
        shufflenet_v2_x1_5 as ShuffleNetV2_1_5, \
        shufflenet_v2_x2_0 as ShuffleNetV2_2
    from .squeezenet import squeezenet1_0 as SqueezeNet_1, \
        squeezenet1_1 as SqueezeNet_1_1
    from .vgg import vgg11 as VGG11, vgg11_bn as VGG11_BN, \
        vgg13 as VGG13, vgg13_bn as VGG13_BN, vgg16 as VGG16, \
        vgg16_bn as VGG16_BN, vgg19 as VGG19, vgg19_bn as VGG19_BN
    from .vgg_cifar import VGG as VGGCIFAR
    from .efficientnet.efficientnet import EfficientNet
    from .efficientnet_cifar import EfficientNetBuild as EfficientNetCIFAR
    from .densenet_cifar import densenet_cifar as DenseNet121CIFAR
else:
    from models.alexnet import alexnet as AlexNet
    from models.densenet import densenet201 as DenseNet201, \
        densenet169 as DenseNet169, densenet161 as DenseNet161,\
        densenet121 as DenseNet121
    from models.googlenet import googlenet as GoogLeNet
    from models.inception import inception_v3 as InceptionV3
    from models.mnasnet import mnasnet0_5 as MNASNet_0_5,\
        mnasnet0_75 as MNASNet_0_75, mnasnet1_0 as MNASNet_1,\
        mnasnet1_3 as MNASNet_1_3
    from models.mobilenet import mobilenet_v2 as MobileNetV2
    from models.mobilenet_cifar import MobileNetV2 as MobileNetV2CIFAR
    from models.senet import SENet18 as SENet18CIFAR
    from models.shufflenetv2_cifar import ShuffleNetV2 as ShuffleNetV2CIFAR
    from models.resnet import resnet18 as ResNet18, resnet34 as ResNet34, \
        resnet50 as ResNet50, resnet101 as ResNet101, resnet152 as ResNet152, \
        resnext50_32x4d as ResNeXt50, resnext101_32x8d as ResNeXt101, \
        wide_resnet50_2 as WideResNet50, wide_resnet101_2 as WideResNet101
    from models.resnet_cifar import ResNet34 as ResNet34CIFAR,\
        ResNet18 as ResNet18CIFAR, ResNet101 as ResNet101CIFAR, ResNet50 as ResNet50CIFAR
    from models.resnext_cifar import ResNeXt29_1x64d as ResNeXtCIFAR
    from models.shufflenetv2 import shufflenet_v2_x0_5 as ShuffleNetV2_0_5, \
        shufflenet_v2_x1_0 as ShuffleNetV2_1, \
        shufflenet_v2_x1_5 as ShuffleNetV2_1_5, \
        shufflenet_v2_x2_0 as ShuffleNetV2_2
    from models.squeezenet import squeezenet1_0 as SqueezeNet_1, \
        squeezenet1_1 as SqueezeNet_1_1
    from models.vgg import vgg11 as VGG11, vgg11_bn as VGG11_BN, \
        vgg13 as VGG13, vgg13_bn as VGG13_BN, vgg16 as VGG16, \
        vgg16_bn as VGG16_BN, vgg19 as VGG19, vgg19_bn as VGG19_BN
    from models.vgg_cifar import VGG as VGGCIFAR
    from models.efficientnet.efficientnet import EfficientNet
    from models.efficientnet_cifar import EfficientNetBuild as EfficientNetCIFAR
    from models.densenet_cifar import densenet_cifar as DenseNet121CIFAR


def get_network(name: str, num_classes: int) -> None:
    return \
        AlexNet(
            num_classes=num_classes) if name == 'AlexNet' else\
        DenseNet201(
            num_classes=num_classes) if name == 'DenseNet201' else\
        DenseNet169(
            num_classes=num_classes) if name == 'DenseNet169' else\
        DenseNet161(
            num_classes=num_classes) if name == 'DenseNet161' else\
        DenseNet121(
            num_classes=num_classes) if name == 'DenseNet121' else\
        DenseNet121CIFAR(
            num_classes=num_classes) if name == 'DenseNet121CIFAR' else\
        GoogLeNet(
            num_classes=num_classes) if name == 'GoogLeNet' else\
        InceptionV3(
            num_classes=num_classes) if name == 'InceptionV3' else\
        MNASNet_0_5(
            num_classes=num_classes) if name == 'MNASNet_0_5' else\
        MNASNet_0_75(
            num_classes=num_classes) if name == 'MNASNet_0_75' else\
        MNASNet_1(
            num_classes=num_classes) if name == 'MNASNet_1' else\
        MNASNet_1_3(
            num_classes=num_classes) if name == 'MNASNet_1_3' else\
        MobileNetV2(
            num_classes=num_classes) if name == 'MobileNetV2' else\
        MobileNetV2CIFAR(
            num_classes=num_classes) if name == 'MobileNetV2CIFAR' else\
        SENet18CIFAR(
            num_classes=num_classes) if name == 'SENet18CIFAR' else\
        ShuffleNetV2CIFAR(
            net_size=1.5, num_classes=num_classes) if name == 'ShuffleNetV2CIFAR' else\
        ResNet18(
            num_classes=num_classes) if name == 'ResNet18' else\
        ResNet34(
            num_classes=num_classes) if name == 'ResNet34' else\
        ResNet34CIFAR(
            num_classes=num_classes) if name == 'ResNet34CIFAR' else\
        ResNet50CIFAR(
            num_classes=num_classes) if name == 'ResNet50CIFAR' else\
        ResNet101CIFAR(
            num_classes=num_classes) if name == 'ResNet101CIFAR' else\
        ResNet18CIFAR(
            num_classes=num_classes) if name == 'ResNet18CIFAR' else\
        ResNet50(
            num_classes=num_classes) if name == 'ResNet50' else\
        ResNet101(
            num_classes=num_classes) if name == 'ResNet101' else\
        ResNet152(
            num_classes=num_classes) if name == 'ResNet152' else\
        ResNeXt50(
            num_classes=num_classes) if name == 'ResNext50' else\
        ResNeXtCIFAR(
            num_classes=num_classes) if name == 'ResNeXtCIFAR' else\
        ResNeXt101(
            num_classes=num_classes) if name == 'ResNext101' else\
        WideResNet50(
            num_classes=num_classes) if name == 'WideResNet50' else\
        WideResNet101(
            num_classes=num_classes) if name == 'WideResNet101' else\
        ShuffleNetV2_0_5(
            num_classes=num_classes) if name == 'ShuffleNetV2_0_5' else\
        ShuffleNetV2_1(
            num_classes=num_classes) if name == 'ShuffleNetV2_1' else\
        ShuffleNetV2_1_5(
            num_classes=num_classes) if name == 'ShuffleNetV2_1_5' else\
        ShuffleNetV2_2(
            num_classes=num_classes) if name == 'ShuffleNetV2_2' else\
        SqueezeNet_1(
            num_classes=num_classes) if name == 'SqueezeNet_1' else\
        SqueezeNet_1_1(
            num_classes=num_classes) if name == 'SqueezeNet_1_1' else\
        VGG11(
            num_classes=num_classes) if name == 'VGG11' else\
        VGG11_BN(
            num_classes=num_classes) if name == 'VGG11_BN' else\
        VGG13(
            num_classes=num_classes) if name == 'VGG13' else\
        VGG13_BN(
            num_classes=num_classes) if name == 'VGG13_BN' else\
        VGG16(
            num_classes=num_classes) if name == 'VGG16' else\
        VGG16_BN(
            num_classes=num_classes) if name == 'VGG16_BN' else\
        VGG19(
            num_classes=num_classes) if name == 'VGG19' else\
        VGG19_BN(
            num_classes=num_classes) if name == 'VGG19_BN' else \
        VGGCIFAR('VGG16',
                 num_classes=num_classes) if name == 'VGG16CIFAR' else \
        EfficientNet.from_name(model_name='-'.join(['b' + s.lower() for s in
                                                    name.split('B') if s])[1:],
                               num_classes=num_classes) if 'EfficientNet' in name and \
        'CIFAR' not in name else \
        EfficientNetCIFAR(model_name='-'.join(['b' + s.lower() for s in
                                               name.replace("CIFAR", "").split('B') if s])[1:],
                          num_classes=num_classes) if 'EfficientNet' in \
        name and 'CIFAR' in name else \
        None
