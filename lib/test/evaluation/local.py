from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # 必需：结果和模型路径
    settings.results_path = r'E:\Pycharm_Project\Algorithm_Challenge_Competition\Vision_Geometry_Semantic\UnTrack_Joint_T_P_search\output\test\tracking_results'
    settings.network_path = r'E:\Pycharm_Project\Algorithm_Challenge_Competition\Vision_Geometry_Semantic\UnTrack_Joint_T_P_search\output\test\networks'
    settings.prj_dir = r"E:\Pycharm_Project\Algorithm_Challenge_Competition\Vision_Geometry_Semantic\UnTrack_Joint_T_P_search"

    # 你的自定义数据集路径（关键！）
    settings.my_data_path = r"E:\Artificial_intelligence\multiple_test_next"
    settings.save_dir = r"E:\Pycharm_Project\Algorithm_Challenge_Competition\Vision_Geometry_Semantic\UnTrack_Joint_T_P_search\output"

    # 可选：如果你要画图
    settings.result_plot_path = r'E:\Pycharm_Project\Algorithm_Challenge_Competition\Vision_Geometry_Semantic\UnTrack_Joint_T_P_search\output\test\result_plots'

    return settings