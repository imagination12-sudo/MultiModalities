import os
import sys
import torch

import os
import warnings
from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore",
                        category=UnsupportedFieldAttributeWarning)
warnings.filterwarnings(
    "ignore",
    message=".*An output with one or more elements was resized.*"
)


# === 动态添加项目根目录到 Python 路径 ===
# 假设本脚本位于 project_root/lib/train/ 或 project_root/ 下
# 我们向上查找直到找到包含 'experiments' 和 'lib' 的目录作为项目根
def find_project_root(current_path, max_depth=5):
    for _ in range(max_depth):
        if os.path.exists(os.path.join(current_path, 'experiments')) and \
           os.path.exists(os.path.join(current_path, 'lib')):
            return current_path
        parent = os.path.dirname(current_path)
        if parent == current_path:  # 到达根目录
            break
        current_path = parent
    raise RuntimeError("无法自动定位项目根目录，请确保脚本在项目结构内运行。")

# 获取当前脚本所在目录，并推断项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
prj_dir = find_project_root(script_dir)
sys.path.insert(0, prj_dir)
# print(f"{prj_dir}===============================================")

# === 导入项目模块 ===
from lib.train.admin.settings import Settings
from lib.train.train_script import run

# print("Current working directory:", os.getcwd())
# print("Project root directory:", prj_dir)

# === 配置训练参数 ===
settings = Settings()
settings.script_name = 'untrack'
settings.config_name = 'deep_rgbx'  # 对应 experiments/untrack/deep_rgbx.yaml
settings.local_rank = -1           # 单卡训练
settings.save_dir = os.path.join(prj_dir, 'output')  # 统一使用项目根下的 output
settings.project_path = 'train/{}/{}'.format(settings.script_name,  settings.config_name)  # 用于日志子目录等（根据框架约定）
settings.distill = None
settings.script_teacher = None
settings.config_teacher = None
settings.cudnn_benchmark = True


# 自动构建配置文件路径
cfg_path = os.path.join(prj_dir, 'experiments', settings.script_name, f'{settings.config_name}.yaml')
if not os.path.isfile(cfg_path):
    raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
settings.cfg_file = cfg_path

# 可选设置
settings.use_wandb = False      # 实验跟踪服务
settings.use_lmdb = False       # 加速速度读取

if __name__ == '__main__':
    # 启动训练
    run(settings)