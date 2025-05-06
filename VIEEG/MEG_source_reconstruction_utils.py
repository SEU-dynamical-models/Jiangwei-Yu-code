"""
Util functions for MEG source reconstruction.

# tags: beamformers, cognitive tasks, volume, source reconstruction
# Author: Miao Cao
# Date: 20 Jul. 2021
# Email: miao.cao@hotmail.com/miao.cao@pku.edu.cn
# Location: Yanyuan, Peking University, Beijing
"""

import pprint
import numpy as np
import mne
from mne.io.constants import FIFF
from mne.surface import _CheckInside
from mne.utils import logger
from mne.transforms import (invert_transform, apply_trans)

def beamform_time_series(epochs, fwd, recon_params, raw_cov=None):
    '''
    This function applies inverse modelling to
    sensor signals using beamformers implemented
    by MNE-Python (Version 1.3.0)
    https://mne.tools/stable/index.html

    Author: Miao Cao
    Email: miao.cao@hotmail.com
    
    Parameters
    ----------
        epoch: mne.Epochs object
            Epochs (mne.Epochs object), containing segmented signals
        fwd: mne.Forward object
            Forward (mne.Forward object) model, containing all information
            including head model geometries to construct a forward solution
        recon_params: dict
        

    Returns
    -------
        list_src_ts: list
            List of reconstructed source spaces
    '''
    list_src_ts                = None
    filters                    = None
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
                                    "info"            : epochs.info,
                                    "forward"         : fwd,
                                    "data_cov"        : None,
                                    "reg"             : recon_params["regularization"],     # reg is vital for beamformers spatial resolutions
                                    "noise_cov"       : None,
                                    "label"           : None,
                                    "pick_ori"        : recon_params["pick_ori"],           # max-power or vector
                                    "rank"            : recon_params["lcmv_rank"],
                                    "weight_norm"     : None,
                                    "reduce_rank"     : False,
                                    "depth"           : None,                               # do not use depth correction
                                    "inversion"       : recon_params["beamformer_inversion"],
                                    "verbose"         : True,}

        # 5.2.a.2 generate data covariance matrix from epoch
        data_cov_method = recon_params['cov_method']
        data_tmin, data_tmax = recon_params['data_cov_tmin'], recon_params['data_cov_tmax']
        beamforming_parameter_dict["data_cov"] = mne.compute_covariance(
                                                                        epochs            = epochs,
                                                                        keep_sample_mean  = True,
                                                                        tmin              = data_tmin,
                                                                        tmax              = data_tmax,
                                                                        projs             = None,
                                                                        method            = data_cov_method,
                                                                        method_params     = None,
                                                                        cv                = 3,
                                                                        scalings          = None,
                                                                        n_jobs            = 12,
                                                                        return_estimators = False,
                                                                        on_mismatch       = "raise",
                                                                        rank              = recon_params['cov_rank'],
                                                                        verbose           = True,)
        # 5.2.a.3 generate noise covariance matrix from either pre-event data or raw_cov (pre-computed)
        noise_cov_method = recon_params['cov_method']
        noise_tmin, noise_tmax = recon_params['noise_cov_tmin'], recon_params['noise_cov_tmax']
        if raw_cov == None:
            if len(epochs) == 1 and epochs.times[0] >= -10:
                noise_tmin, noise_tmax = None, -1
            elif len(epochs) == 1 and epochs.times[0] < -10:
                noise_tmin, noise_tmax = None, -3
            beamforming_parameter_dict["noise_cov"] = mne.compute_covariance(
                                                                            epochs            = epochs,
                                                                            keep_sample_mean  = True,
                                                                            tmin              = noise_tmin,
                                                                            tmax              = noise_tmax,
                                                                            projs             = None,
                                                                            method            = noise_cov_method,
                                                                            method_params     = None,
                                                                            cv                = 3,
                                                                            scalings          = None,
                                                                            n_jobs            = 12,
                                                                            return_estimators = False,
                                                                            on_mismatch       = "raise",
                                                                            rank              = recon_params['cov_rank'],
                                                                            verbose           = True,)
        else:
            noise_tmin, noise_tmax = "start raw data", "end raw data"
            beamforming_parameter_dict["noise_cov"] = raw_cov

        # 5.2.a.4 make inverse operate using beamforming
        filters = mne.beamformer.make_lcmv(
                                            info        = beamforming_parameter_dict["info"],
                                            forward     = beamforming_parameter_dict["forward"],
                                            data_cov    = beamforming_parameter_dict["data_cov"],
                                            reg         = beamforming_parameter_dict["reg"],
                                            noise_cov   = beamforming_parameter_dict["noise_cov"],
                                            label       = beamforming_parameter_dict["label"],
                                            pick_ori    = beamforming_parameter_dict["pick_ori"],
                                            rank        = beamforming_parameter_dict["rank"],
                                            weight_norm = beamforming_parameter_dict["weight_norm"],
                                            reduce_rank = beamforming_parameter_dict["reduce_rank"],
                                            depth       = beamforming_parameter_dict["depth"],
                                            inversion   = beamforming_parameter_dict["inversion"],
                                            verbose     = beamforming_parameter_dict["verbose"],)

        # 5.3.1 apply inverse operator to IED epochs
        # 5.3.1.a use lcmv
        evoked = epochs.average()
        list_src_ts = mne.beamformer.apply_lcmv(evoked  = evoked,
                                                filters = filters,
                                                verbose = True,)

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

def exclude_sources_using_fsaseg(src, aseg_fname):
    from mne._freesurfer import _read_mri_info, _reorient_image
    include_label_dict = dict(
                            cerebral_cortex  = [3, 42],
                            thalamus         = [10, 49],
                            caudate          = [11, 50],
                            putamen          = [12, 51],
                            pallidum         = [13, 52],
                            hippocampus      = [17, 53],
                            amygdala         = [18, 54],
                            accumbens        = [26, 58],
                            ventralDC        = [28, 60],
                            choroid_plexus   = [31, 63],
                            cingulate_cortex = [251, 252, 253, 254, 255],)
    val_list = np.sort([i for row in include_label_dict.values() for i in row])
    # aseg_img = nib.load(aseg_fname)
    # aseg_map = aseg_img.get_fdata()
    # inv_affine = np.linalg.inv(aseg_img.affine)

    _, _, _, _, _, nim = _read_mri_info(aseg_fname, units='mm', return_img=True)

    data, rasvox_mri_t = _reorient_image(nim)
    mri_rasvox_t = np.linalg.inv(rasvox_mri_t)

    list_print = []
    for vert in src[0]['vertno']:
        ras_coords_mm = 1000 * src[0]['rr'][vert]
        vox_inds = np.round(apply_trans(mri_rasvox_t, ras_coords_mm)).astype(int)
        aseg_val = int(data[vox_inds[0], vox_inds[1], vox_inds[2]])
        if aseg_val not in val_list:
            # str_print = f"vertno:{vert} coords:{ras_coords_mm} vox_ind:{vox_inds} aseg_val:{aseg_val}"
            # list_print.append(str_print)
        # else:
            # here, update src data
            src[0]['inuse'][vert] = False
            src[0]['vertno']      = np.delete(src[0]['vertno'], np.where(src[0]['vertno'] == vert))
    
    # last, update number of vertices in use
    src[0]['nuse'] = src[0]['vertno'].shape[0]
    return src

def _exclude_sources_within_white_matter(src, surf, limit=0.0, mri_head_t=None, n_jobs=6,
                        verbose=True):
    """
    Remove all (volume) source space points inside the 
    surface boundary provided by surf parameter as dict.


    Adapted by Miao Cao using mne-python version 1.0.3

    Author: Miao Cao
    Email: miao.cao@hotmail.com
    """
    if src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD and mri_head_t == None:
        raise RuntimeError('Source spaces are in head coordinates and no '
                            'coordinate transform was provided!')

    # How close are the source points to the surface?
    out_str = 'Source spaces are in '
    if src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
        inv_trans = invert_transform(mri_head_t)
        out_str += 'head coordinates.'
    elif src[0]['coord_frame'] == FIFF.FIFFV_COORD_MRI:
        out_str += 'MRI coordinates.'
    else:
        out_str += 'unknown (%d) coordinates.' % src[0]['coord_frame']
    logger.info(out_str)
    out_str = 'Checking that the sources are inside the surface'
    if limit > 0.0:
        out_str += ' and at least %6.1f mm away' % (limit)
    logger.info(out_str + ' (will take a few...)')

    # fit a sphere to a surf quickly
    check_inside = _CheckInside(surf)

    # Check that the source is inside surface (often the inner skull)
    for s in src:
        vertno = np.where(s['inuse'])[0]  # can't trust s['vertno'] this deep
        # Convert all points here first to save time
        r1s = s['rr'][vertno]
        if s['coord_frame'] == FIFF.FIFFV_COORD_HEAD:
            r1s = apply_trans(inv_trans['trans'], r1s)

        inside = check_inside(r1s, n_jobs)
        omit_outside = (~inside).sum()

        # vectorized nearest using BallTree (or cdist)
        omit_limit = 0
        if limit > 0.0:
            # only check "inside" points
            idx = np.where(inside)[0]
            check_r1s = r1s[idx]
            if check_inside.inner_r is not None:
                # ... and those that are at least inner_sphere + limit away
                mask = (np.linalg.norm(check_r1s - check_inside.cm, axis=-1) >=
                        check_inside.inner_r - limit / 1000.)
                idx = idx[mask]
                check_r1s = check_r1s[mask]
            dists = _compute_nearest(
                surf['rr'], check_r1s, return_dists=True, method='cKDTree')[1]
            close = (dists < limit / 1000.0)
            omit_limit = np.sum(close)
            inside[idx[close]] = False
        # s['inuse'][vertno[~inside]] = False
        # how about we set this to be True here?
        inuse_list = deepcopy(s['inuse'])
        inuse_list[:] = False
        inuse_list[vertno[~inside]] = True
        s['inuse'] = np.logical_and(s['inuse'], inuse_list)
        del vertno
        s['nuse'] = omit_outside + omit_limit # here, sources outside the surface are kept
        s['vertno'] = np.where(s['inuse'])[0]

        if omit_outside > 0:
            extras = [omit_outside]
            extras += ['s', 'they are'] if omit_outside > 1 else ['', 'it is']
            logger.info('    %d source space point%s omitted because %s '
                        'outside the inner skull surface.' % tuple(extras))
        if omit_limit > 0:
            extras = [omit_limit]
            extras += ['s'] if omit_outside > 1 else ['']
            extras += [limit]
            logger.info('    %d source space point%s omitted because of the '
                        '%6.1f-mm distance limit.' % tuple(extras))
        # Adjust the patch inds as well if necessary
        if omit_limit + omit_outside > 0:
            _adjust_patch_info(s)
    return check_inside

def read_sensor_geometry(pom_fname, 
                            sensor_name_start     = "REMARK_LIST START_LIST",
                            sensor_name_end       = "REMARK_LIST END_LIST",
                            sensor_location_start = "LOCATION_LIST START_LIST",
                            sensor_location_end   = "LOCATION_LIST END_LIST",
                            num_sensor_marker     = "ListNrRows"):
    import re
    import numpy as np

    with open(pom_fname, 'rt', encoding='utf-8-sig') as pom_file:
        pom_lines = pom_file.readlines()

    sensor_num_list = [int(re.findall("\d+", line)[0]) for line in pom_lines if re.search(f"{num_sensor_marker} *", line)]
    sensor_num      = sensor_num_list[0]


    sensor_locations = []
    sensor_names     = []
    pattern = ".*\d*\.\d+|.*\d+"
    for ind, line in enumerate(pom_lines):
        # get sensor locations
        if re.search(f"{sensor_location_start} *", line):
            for sensor in np.arange(1, sensor_num + 1):
                sensor_line = pom_lines[ind + sensor].replace('\n', '').replace(' ', '')
                sensor_locations.append([float(item) for item in sensor_line.split("\t")])
        # ge sensor names
        if re.search(f"{sensor_name_start} *", line):
            for sensor in np.arange(1, sensor_num + 1):
                sensor_line = pom_lines[ind + sensor].replace('\n', '')
                sensor_names.append(sensor_line)

    sensor_geo_info = {}         
    for ind, sensor_name in enumerate(sensor_names):
        sensor_geo_info[sensor_name] = np.array(sensor_locations[ind]) * 0.001
    
    return sensor_geo_info

def compute_empty_room_recording_cov(erm_raw, data_cov_method='auto', cov_rank=None):
    erm_cov = mne.compute_raw_covariance(
                                    raw     = erm_raw,
                                    picks   = 'meg',
                                    method  = data_cov_method,
                                    n_jobs  = 24,
                                    rank    = cov_rank,
                                    verbose = True,            )
    
    return erm_cov

def read_ViEEG_coordinates(fname_vg_coords):
    '''
    This function reads the text file of ViEEG coordinates

    Parameters
    ----------
        fname_vg_coords: String
            A string, specifying path and file name of the text file containing coordinate
            and labels of ViEEG electrodes
    Returns
    -------
        no return
    '''
    print('Reading virtual grid configuration from ' + fname_vg_coords)

    vg_file = open(fname_vg_coords, 'r', encoding='utf-8-sig')
    vg_file_lines = vg_file.readlines()
    vg_coordinates = []
    vg_elect_labels = []
    num_vg_elects = int(vg_file_lines[0])
    for read_cursor in np.arange(1, num_vg_elects + 1):
        line = vg_file_lines[read_cursor]
        line_list = line.strip('\n').split('\t')
        # Read virtual grid coordinates from pom file generated from Curry
        vg_coordinates.append([float(num) for num in line_list])

    for read_cursor in np.arange(num_vg_elects + 1, num_vg_elects * 2 + 1):
        line = vg_file_lines[read_cursor]
        line.strip('\n').split('\t')
        vg_elect_labels.append(line)
    return np.array(vg_coordinates), np.array(vg_elect_labels)

def read_ViEEG_CurryRes_coordinates(fname_vg_coords):
    '''
    This function reads the text file of ViEEG coordinates

    Parameters
    ----------
        fname_vg_coords: String
            A string, specifying path and file name of the text file containing coordinate
            and labels of ViEEG electrodes
    Returns
    -------
        no return
    '''
    print('Reading ViEEG configuration from CURRY RES file:' + fname_vg_coords)

    vg_file = open(fname_vg_coords, 'r', encoding='utf-8-sig')
    vg_file_lines = vg_file.readlines()
    vg_coordinates = []
    vg_elect_labels = []
    num_vg_elects = len(vg_file_lines)
    for line in vg_file_lines:
        line_list = line.strip('\n').split('\t')
        coord_3d = line_list[1:4]
        
        # Read virtual grid coordinates from CURRY RES file generated from Brainstorm
        vg_coordinates.append([float(num) for num in coord_3d])
        vg_elect_labels.append(line_list[-1])

    return np.array(vg_coordinates), np.array(vg_elect_labels)