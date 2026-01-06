import importlib    # 动态模块导入文件
import os
from collections import OrderedDict


def create_default_local_file():
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': empty_str,
        'tensorboard_dir': 'self.workspace_dir + \'/tensorboard/\'',
        'pretrained_networks': 'self.workspace_dir + \'/pretrained_networks/\'',
        'lasot_dir': empty_str,
        'got10k_dir': empty_str,
        'trackingnet_dir': empty_str,
        'coco_dir': empty_str,
        'lvis_dir': empty_str,
        'sbd_dir': empty_str,
        'imagenet_dir': empty_str,
        'imagenetdet_dir': empty_str,
        'ecssd_dir': empty_str,
        'hkuis_dir': empty_str,
        'msra10k_dir': empty_str,
        'davis_dir': empty_str,
        'youtubevos_dir': empty_str})

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def create_default_local_file_train(workspace_dir, data_dir):
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'workspace_dir': workspace_dir,
        'tensorboard_dir': os.path.join(workspace_dir, 'tensorboard'),    # Directory for tensorboard files.
        'pretrained_networks': os.path.join(workspace_dir, 'pretrained_networks'),
        'got10k_val_dir': os.path.join(data_dir, 'got10k/val'),
        'lasot_lmdb_dir': os.path.join(data_dir, 'lasot_lmdb'),
        'got10k_lmdb_dir': os.path.join(data_dir, 'got10k_lmdb'),
        'trackingnet_lmdb_dir': os.path.join(data_dir, 'trackingnet_lmdb'),
        'coco_lmdb_dir': os.path.join(data_dir, 'coco_lmdb'),
        'coco_dir': os.path.join(data_dir, 'coco'),
        'lasot_dir': os.path.join(data_dir, 'lasot'),
        'got10k_dir': os.path.join(data_dir, 'got10k/train'),
        'trackingnet_dir': os.path.join(data_dir, 'trackingnet'),
        'depthtrack_dir': os.path.join(data_dir, 'depthtrack/train'),
        'lasher_dir': os.path.join(data_dir, 'lasher/trainingset'),
        'visevent_dir': os.path.join(data_dir, 'visevent/train'),
    })

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                if attr_val == empty_str:
                    f.write('        self.{} = {}\n'.format(attr, attr_val))
                else:
                    f.write('        self.{} = \'{}\'\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))


def env_settings():
    # 指定要导入的模块名称
    env_module_name = 'lib.train.admin.local'
    try:
        # 动态尝试导入指定模块
        env_module = importlib.import_module(env_module_name)
        # 如果导入成功， 调用该模块的EnvironmentSettings类并返回实例
        return env_module.EnvironmentSettings()
    except:
        # 导入失败， 通常因为local.py 文件不存在
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        # 创建一个默认的本地配置文件模板
        create_default_local_file()
        # 抛出错误，提示用户取配置local.py文件
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. Then try to run again.'.format(env_file))
