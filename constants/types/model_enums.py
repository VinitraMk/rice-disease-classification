import enum

class Model(str, enum.Enum):
    RNN = 'rnn'
    CNN = 'cnn'
    ALEXNET = 'alexnet'
    RESNET = 'resnet'