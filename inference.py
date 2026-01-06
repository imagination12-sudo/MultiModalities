from lib.test.parameter.untrack import parameters
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.running import run_dataset

# 加载数据集
dataset = get_dataset('mydataset')

# 创建 tracker
trackers = trackerlist(
    name='untrack',        # 对应 lib/test/tracker/my_rgbd_tracker.py
    parameter_name='deep_rgbx',      # 对应 lib/test/parameter/my_rgbd_tracker.py 中的参数
    dataset_name='depthtrack_rgbd',
    run_ids=None
)


if __name__ == '__main__':
    run_dataset(
        trackers=trackers,
        dataset=dataset,
        debug=False,
        threads=4,        # 并行线程数
        num_gpus=1        # 可用 GPU 数
    )