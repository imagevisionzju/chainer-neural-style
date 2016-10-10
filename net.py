import math

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

class VGG(chainer.Chain):
    def __init__(self):
        super(VGG, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, stride=1, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1),

            conv2_1=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1),

            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1),

            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1),

            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1)
        )
        self.mean = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)#bgr mean value:    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
    def preprocess(self, image, input_type='BGR'):
        if input_type == 'RGB':
            image = image[:,:,::-1]
        return np.rollaxis(image - self.mean, 2)

    def __call__(self, x):
        y1 = F.relu(self.conv1_2(F.relu(self.conv1_1(x))))
        h = F.max_pooling_2d(y1, 2, stride=2)
        y2 = F.relu(self.conv2_2(F.relu(self.conv2_1(h))))
        h = F.max_pooling_2d(y2, 2, stride=2)
        y3 = F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(h))))))
        h = F.max_pooling_2d(y3, 2, stride=2)
        y4 = F.relu(self.conv4_3(F.relu(self.conv4_2(F.relu(self.conv4_1(h))))))
        #no need calc of 5_x
        return [y1, y2, y3, y4]

