class EnvironmentSettings:
    def __init__(self):
        # 工作区根目录（模型保存、日志等）
        self.workspace_dir = r'/Vision_Geometry_Semantic/UnTrack_Joint_T_P_search\tracking'

        # TensorBoard 日志目录
        self.tensorboard_dir = r'/Vision_Geometry_Semantic/UnTrack_Joint_T_P_search\tracking\tensorboard'

        # 预训练模型目录
        self.pretrained_networks = r'E:\Pycharm_Project\Algorithm_Challenge_Competition\Vision_Geometry_Semantic\UnTrack_Joint_T_P\pretrained'

        # 训练集绝对路径
        self.depthtrack_dir_train = r"E:\Artificial_intelligence\mulmodalities_train\TrainSet"
        # 验证集绝对路径
        self.depthtrack_dir_val = r"E:\Artificial_intelligence\mulmodalities_train\ValidationSet"
