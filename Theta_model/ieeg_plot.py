#功能：查看已预处理的ieeg数据集并画图
#作者：余江伟
#时间：2024/5/21
#联系方式：15300593720

import os
import os.path as op
import matplotlib.pyplot as plt
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
dataset = "ummc"
sub = "9"
bids_root = "D:\qq文件\交接代码\数据集\data\processed/" + dataset +"/sub"+ sub +"\BIDS_dataset"
datatype = 'eeg'
#session = '01'
session = 'presurgery'
task = 'GC'
suffix = 'eeg'
#subject = dataset + sub
subject = dataset+ "00" + sub
bids_path = BIDSPath(subject=subject, session=session, task=task, run='01',
                     suffix=suffix, datatype=datatype, root=bids_root)
print(bids_path)
raw = read_raw_bids(bids_path=bids_path, verbose=False)
raw.plot(scalings=1,n_channels=200,duration=20)
plt.tight_layout()
savepath = "D:\qq文件\交接代码\数据集\data\processed/20patients_plt/" + subject +"_run01_20seconds.png"
plt.savefig(savepath)
plt.show()
