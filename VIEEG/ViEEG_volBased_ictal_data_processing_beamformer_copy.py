# -*- coding: utf-8 -*-
"""
ViEEG volume-based source reconstruction of ictal dynamics.

# tags: beamformers, ictal source reconstruction, vieeg
# Author: Miao Cao
# Date: 18 Apr. 2021
# Email: caomiao89@gmail.com
"""

import os
import re
import pprint
import datetime
from pathlib import Path

import mne
import mne_bids
import yaml
import numpy as np
import hdf5storage as hdf
import matplotlib.pyplot as plt

from MEG_source_reconstruction_utils import beamform_time_series, read_ViEEG_CurryRes_coordinates

def _beamform_time_series(epochs, fwd, recon_params, raw_cov=None):
    '''
    This function applies inverse modelling to
    sensor signals using beamformers implemented
    by MNE-Python (Version 0.22.1)
    https://mne.tools/stable/index.html

    Parameters
    ----------
        epoch: mne.Epochs object
            Epochs (mne.Epochs object), containing segmented signals
        fwd: mne.Forward object
            Forward (mne.Forward object) model, containing all information
            including head model geometries to construct a forward solution

    Returns
    -------
        list_src_ts: list
            List of reconstructed source spaces
    '''
    list_src_ts = None
    filters = None
    beamforming_parameter_dict = None
    if 'beamforming_method' in recon_params.keys():
        beamforming_method = recon_params['beamforming_method']
    else:
        beamforming_method = "lcmv"
    # make filters according to different beamforming methods
    if beamforming_method == "lcmv":
        # 5.2.a.1 define parameters
        beamforming_parameter_dict = {
            "beamforming_name": beamforming_method,
            "info": epochs.info,
            "forward": fwd,
            "data_cov": None,
            "reg": 0.01,  # reg appears to be very critical for spatial resolution
            "noise_cov": None,
            "label": None,
            "pick_ori": 'max-power',
            "rank": None,
            "weight_norm": None,
            "reduce_rank": True,
            "depth": None,  # not use depth correction for now
            "inversion": 'single',
            "verbose": True,
        }

        # 5.2.a.2 generate data covariance matrix from epoch
        data_cov_method = recon_params['cov_method'] #"auto"
        data_tmin, data_tmax = recon_params['data_cov_tmin'], recon_params['data_cov_tmax']
        beamforming_parameter_dict["data_cov"] = mne.compute_covariance(
            epochs=epochs,
            keep_sample_mean=True,
            tmin=data_tmin,
            tmax=data_tmax,
            projs=None,
            method=data_cov_method,
            method_params=None,
            cv=3,
            scalings=None,
            n_jobs=12,
            return_estimators=False,
            on_mismatch="raise",
            rank=None,
            verbose=True,
        )
        # 5.2.a.3 generate noise covariance matrix from either pre-event data or raw_cov (pre-computed)
        noise_cov_method = recon_params['cov_method'] #"auto"
        noise_tmin, noise_tmax = recon_params['noise_cov_tmin'], recon_params['noise_cov_tmax']
        if raw_cov is None:
            noise_tmin, noise_tmax = None, -30
            beamforming_parameter_dict["noise_cov"] = mne.compute_covariance(
                epochs=epochs,
                keep_sample_mean=True,
                tmin=noise_tmin,
                tmax=noise_tmax,
                projs=None,
                method=noise_cov_method,
                method_params=None,
                cv=3,
                scalings=None,
                n_jobs=12,
                return_estimators=False,
                on_mismatch="raise",
                rank=None,
                verbose=True,
            )
        else:
            noise_tmin, noise_tmax = "start raw data", "end raw data"
            beamforming_parameter_dict["noise_cov"] = raw_cov

        # 5.2.a.4 make inverse operate using beamforming
        filters = mne.beamformer.make_lcmv(
            info=beamforming_parameter_dict["info"],
            forward=beamforming_parameter_dict["forward"],
            data_cov=beamforming_parameter_dict["data_cov"],
            reg=beamforming_parameter_dict["reg"],
            noise_cov=beamforming_parameter_dict["noise_cov"],
            label=beamforming_parameter_dict["label"],
            pick_ori=beamforming_parameter_dict["pick_ori"],
            rank=beamforming_parameter_dict["rank"],
            weight_norm=beamforming_parameter_dict["weight_norm"],
            reduce_rank=beamforming_parameter_dict["reduce_rank"],
            depth=beamforming_parameter_dict["depth"],
            verbose=beamforming_parameter_dict["verbose"],
        )

        # 5.3.1 apply inverse operator to IED epochs
        # 5.3.1.a use lcmv
        list_src_ts = mne.beamformer.apply_lcmv_epochs(
            epochs=epochs,
            filters=filters,
            # max_ori_out='signed',
            return_generator=False,
            verbose=True,
        )

        # list_src_ts = mne.beamformer.apply_lcmv_epochs(epochs=epoch,
        #                                                filters=filters,
        #                                                max_ori_out='signed',
        #                                                return_generator=False,
        #                                                verbose=True)
        beamforming_parameter_dict["info"] = "raw_meg_data_info is not writable here."
        beamforming_parameter_dict["forward"] = pprint.pformat(
            list(fwd.values()))
        beamforming_parameter_dict["data_cov"] = (
            "covariance matrix estimated from seizure epoch from "
            + str(data_tmin)
            + "s to "
            + str(data_tmax)
            + "s"
            + " using method "
            + data_cov_method
        )
        beamforming_parameter_dict["noise_cov"] = (
            "covariance matrix estimated from seizure epoch from "
            + str(noise_tmin)
            + "s to "
            + str(noise_tmax)
            + "s"
            + " using method "
            + noise_cov_method
        )
    return list_src_ts, beamforming_parameter_dict

print(__doc__)
# Raw data path. This path is pointing to 4KMEG data repository. It's
# normally a different location from analysis scripts. (we make this
# constant for now)
# yml_config_folder = Path(os.path.abspath(__file__)).parent.parent
yml_config_folder = Path(os.path.abspath(__file__)).parent
config_fname = "D:\qq文件\pythonProject\ViEEG_ictal_beamformer_copy.yaml"
with open(config_fname, encoding= "UTF-8") as file:
    config = yaml.safe_load(file)

freesurfer_subjects_dir = config["freesurfer_subjects_dir"]
base_data_folder_path   = config["base_data_folder_path"] # this is on CMR workstation
# base_data_folder_path   = "/Users/miaoc/Documents/Clinical Neurosciences/Data/4KMEG" # this is on MC MBP
raw_data_path           = base_data_folder_path + "/RawData"
preprocessed_data_path  = config["preprocessed_data_path"]
post_analysis_data_path = config["post_analysis_data_path"]
ictal_events_file_path  = config["ictal_events_file_path"]
ViEEG_config_data_path  = config["ViEEG_config_data_path"]

# patient list
patient_list = config["patient_list"]
ViEEG_config_fname_list = config["ViEEG_config_fname_list"]

# choose patient
patient_ID = patient_list[8]

# default MEG sample frequency
MEG_sfreq = config["MEG_sfreq"]
switch_check_virtual_gride_coordinate_locations = True
switch_save_ictal_epoch_raw_data = True
switch_save_source_signals_mat = True
# ---------------------------------------
# 5.1 make forward solution
# ---------------------------------------
# -bem-sol.fif BEM solution

# if forward model has been made by this or other pipelines, switch this off
# True when need to make a forward model
# read in virtual grid configurations
BEM_method      = config["BEM_method"]
bem_spacing     = config["bem_spacing"]
fname_trans     = f"{freesurfer_subjects_dir}/{patient_ID}/bem/{patient_ID}-trans.fif"
fname_vg_coords = f"{ViEEG_config_data_path}/{patient_ID}/{ViEEG_config_fname_list[patient_ID]}"

vg_coordinates, vg_labels = read_ViEEG_CurryRes_coordinates(fname_vg_coords)
vg_labels = np.array([label.strip('\n') for label in vg_labels])

fwd = None
fname_bem = f"{freesurfer_subjects_dir}/{patient_ID}/bem/{patient_ID}-bem.fif"
fname_bem_sol = f"{freesurfer_subjects_dir}/{patient_ID}/bem/{patient_ID}-bem-sol.fif"
pos_rr = np.array([[1, 0, 0]])
pos_nn = np.array([[1, 0, 0]])
vg_coordinates = np.array(vg_coordinates) #* 0.001  # convert to meters
for vg_coord in vg_coordinates:
    pos_rr = np.append(pos_rr, [vg_coord], axis=0)
    # pos_rr = np.append(pos_rr, [vg_coord], axis=0)
    # pos_rr = np.append(pos_rr, [vg_coord], axis=0)
    # pos_nn = np.append(pos_nn, [[1, 0, 0]], axis=0)
    # pos_nn = np.append(pos_nn, [[0, 1, 0]], axis=0)
    pos_nn = np.append(pos_nn, [[0, 0, 1]], axis=0)
pos_rr = np.delete(pos_rr, [0], axis=0)
pos_nn = np.delete(pos_nn, [0], axis=0)
pos    = {"rr": pos_rr, "nn": pos_nn}

vol_src = mne.setup_volume_source_space(
                                        subject          = patient_ID,
                                        pos              = pos,
                                        mri              = None,
                                        bem              = fname_bem,
                                        surface          = None,
                                        mindist          = 0.0,
                                        exclude          = 0.0,
                                        subjects_dir     = freesurfer_subjects_dir,
                                        volume_label     = None,
                                        add_interpolator = False,
                                        verbose          = True,)

# %% read/load ictal events file
EP_fnames = [fname for fname in os.listdir(f"{preprocessed_data_path}/{patient_ID}") if fname.endswith("-epo.fif")]

# ictal_events_dict = {}
# for data_fname in ictal_event_file_list:
#     ictal_event_fname = f"{ictal_events_file_path}/{patient_ID}/{data_fname}"
#     # print(peak_time_data_fname)
#     fif_fname = data_fname[0 : -13] + 'tsss.fif'
#     ictal_events_dict[fif_fname] = mne.read_events(ictal_event_fname)

recon_params                         = {}
recon_params['noise_cov_tmax']       = config["noise_cov_tmax"]
recon_params['noise_cov_tmin']       = config["noise_cov_tmin"]
recon_params['data_cov_tmax']        = config["data_cov_tmax"]
recon_params['data_cov_tmin']        = config["data_cov_tmin"]
recon_params['beamformer_method']    = config["beamformer_method"]
recon_params['beamformer_inversion'] = config["beamformer_inversion"]
recon_params['lcmv_rank']            = config["lcmv_rank"]
recon_params['regularization']       = config["regularization"]
recon_params['cov_method']           = config["cov_method"]
recon_params['cov_rank']             = config["cov_rank"]
recon_params['pick_ori']             = config["beamformer_pick_ori"]
# select ictal events from MEG recordings
ViEEG_ictal_epoch_list = []
# sz_counter = 0
for data_fname in EP_fnames:

    vg_coordinates, vg_labels = read_ViEEG_CurryRes_coordinates(fname_vg_coords)
    vg_coordinates = np.array(vg_coordinates)  # * 0.001  # convert to meters
    for vg_coord in vg_coordinates:
        pos_rr = np.append(pos_rr, [vg_coord], axis=0)
        # pos_rr = np.append(pos_rr, [vg_coord], axis=0)
        # pos_rr = np.append(pos_rr, [vg_coord], axis=0)
        # pos_nn = np.append(pos_nn, [[1, 0, 0]], axis=0)
        # pos_nn = np.append(pos_nn, [[0, 1, 0]], axis=0)
        pos_nn = np.append(pos_nn, [[0, 0, 1]], axis=0)

    EP_ID = re.findall(r"EP_\d*", data_fname)[0].split("_")[1]
    preprocessed_MEG_data_fname = f"{preprocessed_data_path}/{patient_ID}/{data_fname}"
    fname_trans = f"{freesurfer_subjects_dir}/{patient_ID}/bem/trans/{patient_ID}_EP_{EP_ID}_tsss-trans.fif"
    
    MEG_data_preprocessed = mne.read_epochs(fname=preprocessed_MEG_data_fname, preload=True)

    MEG_sfreq = MEG_data_preprocessed.info['sfreq']
    
    if switch_check_virtual_gride_coordinate_locations == True:
        np_vg_coordinates = np.array(
            vg_coordinates)  # convert to meters
        fig1 = mne.viz.plot_alignment(
            info=MEG_data_preprocessed.info,
            #trans=fname_trans,
            subject=patient_ID,
            subjects_dir=freesurfer_subjects_dir,
            meg=False,#false
            dig=True,
            mri_fiducials=False,
            # bem=fname_bem_sol,
            # surfaces=dict(head=0.8, pial=0.8),
            surfaces=dict(pial=0.4, inner_skull=0.3),
            coord_frame="mri",
        )
        # this is plot vieeg electrodes
        pl = fig1.plotter
        actor = pl.add_points(np_vg_coordinates, point_size=30)
        pl.app.exec_()#必须添加此行代码，保证渲染不自动退出

    #     # This is to visualise in Linux using older MNE versions
    #     from mayavi import mlab
    #     mlab.points3d(
    #         np_vg_coordinates[:, 0],
    #         np_vg_coordinates[:, 1],
    #         np_vg_coordinates[:, 2],
    #         mode="2darrow",
    #         line_width=0.5,
    #         figure=fig1,
    #     )
    # continue

    # make the forward solution
    # here read forward model
    
    fwd = mne.make_forward_solution(
        info=MEG_data_preprocessed.info,
        trans=fname_trans,
        src=vol_src,
        bem=fname_bem_sol,
        meg=True,
        eeg=False,
        mindist=0.0,
        ignore_ref=False,
        n_jobs=12, #12
        verbose=True,
    )
    #
    vg_coordinates = vg_coordinates[fwd['src'][0]['inuse'].astype(bool), :]
    vg_labels = vg_labels[fwd['src'][0]['inuse'].astype(bool)]
    #
    # # prepare the ictal event as epochs
    # # epoch the data
    # # tmin = pre_ictal_time
    # # tmax = (event[1, 0] - event[0, 0]) /MEG_sfreq + post_ictal_time
    # # if switch_save_ictal_epoch_raw_data:
    # #     epoch_data_fname = mne_bids.BIDSPath(subject=patient_ID.replace('_', '+'),
    # #                                         # session="MEG",
    # #                                         task="ictal",
    # #                                         acquisition=f"EP{EP_ID}",
    # #                                         run=str(nn_event+1),
    # #                                         processing="raw",
    # #                                         recording="epoch",
    # #                                         space=None,
    # #                                         split=None,
    # #                                         root=post_analysis_data_path,
    # #                                         # suffix=gen_time,
    # #                                         extension=None,
    # #                                         datatype="meg",
    # #                                         check=True,)
    # #     epoch.save(fname=epoch_data_fname.fpath, overwrite=True)
    #
    #
    ViEEG_ictal_epoch, bf_params = beamform_time_series(
        MEG_data_preprocessed, fwd, recon_params)

    vol_src_data_fname = f"{preprocessed_MEG_data_fname[:-8]}_vieeg"
    mne.write_cov(f"{vol_src_data_fname}-noise_cov.fif", bf_params["noise_cov"],overwrite = True)
    mne.write_cov(f"{vol_src_data_fname}-data_cov.fif", bf_params["data_cov"],overwrite = True)
    bf_params.pop("noise_cov")
    bf_params.pop("data_cov")
    # ViEEG_ictal_epoch_list = ViEEG_ictal_epoch_list + ViEEG_ictal_epoch
    # concat ictal data epochs

    params = {}
    # params['pre_ictal_time'] = pre_ictal_time
    # params['post_ictal_time'] = post_ictal_time
    params['meg_data_config'] = config
    params['beamformer_params'] = bf_params
    # ictal_ViEEG_data_mdict = {
    #     'ViEEG_seizure': ViEEG_ictal_epoch[0].data,
    #     'ViEEG_times': ViEEG_ictal_epoch[0].times,
    #     'sfreq': MEG_sfreq,
    #     'params': params,
    #     'generated_time': gen_time,
    # }
    # sz_counter += 1
    # ictal_ViEEG_data_fname = post_analysis_data_path + '/ViEEG_ictal_data/' + patient_ID + '_ictal_ViEEG_data_sz' + str(sz_counter) + '_' + gen_time + '.mat'
    # hdf.savemat(file_name=ictal_ViEEG_data_fname, mdict=ictal_ViEEG_data_mdict)
    # ---------------------------------------
    # 6 visualise source estimates and save source data
    # ---------------------------------------
    # brain_stc = surf_src.plot(subject=patient_ID,
    #                             hemi='both',
    #                             alpha=0.8,
    #                             subjects_dir=freesurfer_subjects_dir,
    #                             brain_kwargs=dict(views='ventral'))

    # Here, start using bids (mne_bids)
    gen_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # vol_src_data_fname = mne_bids.BIDSPath(subject=patient_ID.replace('_', '++'),
    #                                         # session="MEG",
    #                                         task="ictal",
    #                                         acquisition=f"EP{EP_ID}",
    #                                         run=str(nn_event+1),
    #                                         processing="vieeg",
    #                                         recording=None,
    #                                         space=None,
    #                                         split=None,
    #                                         root=post_analysis_data_path,
    #                                         # suffix=gen_time,
    #                                         extension=None,
    #                                         datatype="meg",
    #                                         check=True,)

    #vol_src_data_fname = f"{preprocessed_MEG_data_fname[:-8]}_vieeg"

    # here, save epoch sensor signals
    # save_epoch = epoch.copy().resample(sfreq=100, verbose=True).pick_types(meg=True)
    # save_epoch = epoch.copy().pick_types(meg=True)

    # here, save data as .mat file
    if switch_save_source_signals_mat:
        # ViEEG_vol_src = ViEEG_ictal_epoch[0].resample(sfreq=100, verbose=True)
        ViEEG_vol_src = ViEEG_ictal_epoch
        mdict = {
                'times'             : ViEEG_vol_src.times,
                'sensor_signals'    : np.squeeze(MEG_data_preprocessed.get_data(picks='meg')),
                'sensor_names'      : MEG_data_preprocessed.info['ch_names'],
                'source_data'       : ViEEG_vol_src.data,
                'sfreq'             : ViEEG_vol_src.sfreq,
                'vieeg_coords'      : vg_coordinates,
                'vieeg_names'       : vg_labels,
                'seizure_time_range': [0, MEG_data_preprocessed.times[-1]],
                'gen_time'          : gen_time,
                'params'            : params,                                                  }
        # mat_fname = surf_src_fname + ".mat"
        hdf.savemat(file_name=f"{vol_src_data_fname}.mat", mdict=mdict)
