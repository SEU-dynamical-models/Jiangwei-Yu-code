import numpy as np
import mne
import matplotlib.pyplot as plt
import hdf5storage as hdf

mat_file_path = "D:\qq文件\交接代码\SEU_cohort\SEU_cohort\MEG_ICA\MEG1889\MEG1889_EP_1_sz_1_vieeg.mat"
mat_data = hdf.loadmat(mat_file_path)
mat_info = mne.create_info(mat_data['sensor_names'],mat_data['sfreq'],ch_types="ref_meg")
raw_mat = mne.io.RawArray(mat_data['sensor_signals'],mat_info)
raw_mat.plot(scalings=4e-11, n_channels=30,duration=200)
plt.tight_layout()
plt.show(block=True)

epo_file_path = "D:\qq文件\交接代码\SEU_cohort\SEU_cohort\MEG_ICA\MEG1889\MEG1889_EP_1_sz_1-epo.fif"
epo_raw = mne.read_epochs(epo_file_path)
epo_raw.plot(n_channels=30)
plt.tight_layout()
plt.show(block=True)
