from lib.utils import TensorDict

"""
BaseActor 是一个抽象基类（Abstract Base Class），
它在目标跟踪（或更广泛的深度学习训练）框架中扮演着 “训练逻辑协调者” 的核心角色。它的作用不是直接实现具体功能，
而是定义统一接口和基础行为，让所有具体的 Actor（如 UntrackActor、OSTrackActor 等）遵循一致的规范。
"""

class BaseActor:
    """ Base class for actor. The actor class handles the passing of the data through the network
    and calculation the loss"""
    def __init__(self, net, objective):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        """ Called in each training iteration. Should pass in input data through the network, calculate the loss, and
        return the training stats for the input data
        args:
            data - A TensorDict containing all the necessary data blocks.

        returns:
            loss    - loss for the input data
            stats   - a dict containing detailed losses
        """
        raise NotImplementedError

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)