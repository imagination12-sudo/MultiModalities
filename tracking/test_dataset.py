import
from lib.train.dataset import DepthTrack

ds = DepthTrack(split='train')
print(ds.get_sequence_info(0))  # 查看第一个序列的 bbox
print(ds.get_sequence_info(1))
print(ds.get_sequence_info(2))
print(ds.get_sequence_info(3))
print(ds.get_sequence_info(4))
print(ds.get_sequence_info(5))
print(ds.get_sequence_info(6))
print(ds.get_sequence_info(7))
