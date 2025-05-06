# This is a pipeline for processing structural MR scans (T1, MPRage and FLASH).
# The output from this pipeline is used in source analysis (MNE-Python).
# The target MR scan is from focal epilepsy cohort
# Author: Miao Cao
# Email: miao.cao@unimelb.edu.au

import mne
import mne.bem
import subprocess
import socket

import matplotlib.pyplot as plt
# import scipy
# import scipy.sparse.csgraph  # This is important


# Data paths
# -----------------------------------------------------------
# freesurfer_subjects_dir = "/Users/miaoc/Documents/Clinical Neurosciences/Data/PhD Thesis/Connectivity_Simulations/Freesurfer"
# freesurfer_subjects_dir = "/mnt/process/Miaoc/Data/EEG-MEG/Freesurfer"
# freesurfer_subjects_dir = "/mnt/d/Data/Freesurfer/subjects/"
# freesurfer_subjects_dir = "/mnt/nfs/sanbo_dataset_108/4k_Project/Data_Full/MRI_FS"
# freesurfer_subjects_dir = "/data/miaoc/vvf_meg/sMRI"
# freesurfer_subjects_dir = "/data/miaoc/ictal_cohort/MRI_FS"
# freesurfer_subjects_dir = "/data/miaoc/sleep/MEG/MRI_FS"
freesurfer_subjects_dir = "/Users/miaoc/Desktop/Memory/MRI_FS"

# hostname = socket.gethostname()
# IPAddr = socket.gethostbyname(hostname)
# if IPAddr == "172.25.134.219":
#     freesurfer_subjects_dir = "/data/mcao/projects/freesurfer"
# -----------------------------------------------------------


# Patient list
# -----------------------------------------------------------
# patient_list = ['P037_AB_20141012',
#                 'P061_AE_20151122',
#                 'P007_GS',
#                 'P035_IA',
#                 'P042_KD',
#                 'P048_RL',
#                 'P061_AE',
#                 'P099_IW']  # a list of patient IDs

# # Patient list, a list of patient IDs
# patient_list = ['P012_EM',
#                 'P016_SD',  # * SV's thesis case
#                 'P035_IA',
#                 'P037_AB',  # *, 0143, SV's thesis case
#                 'P040_EW',
#                 'P042_KD',  # * SV's thesis case
#                 'P046_KX',
#                 'P048_RL',
#                 'P049_JC2',
#                 'P061_AE',  # 0143 1913 2133
#                 'P072_TB',  # EPC
#                 'P074_AF',
#                 'P077_TR',  # * SV's thesis case
#                 'P098_SD2',  # * SV's thesis case
#                 'P100_EW2'
#                 ]

# Select a patient and get its patient ID
# patient_ID = patient_list[4]
# patient_ID = "MC_MEFLASH2_20180726"
# patient_ID = "P061_AE"
# patient_ID = "CenZhehang"
# patient_ID = "sub13"
# patient_ID = "P101MG"
patient_ID = "EMF005"

# -----------------------------------------------------------

# Important parameters
# -----------------------------------------------------------
bem_application = "MEG" # "EEG" or "MEG"
bem_spacing = None
# src_spacing = "ico5"  # 4.9 mm, 4098 sources per hemisphere
BEM_method = "ft"
# BEM conductivity. If single value means only inner skull is used and mainly used for MEG; if three values means inner skull, outer skull and outer skin. Three layer BEM is mainly used for EEG.
BEM_conductivity = None
if bem_application == "MEG":
    BEM_conductivity = [0.3]
elif bem_application == "EEG":
    BEM_conductivity = [0.3, 0.006, 0.3]
# path to meFLASH images. Only used when meFLASH sequence is used.
meFLASH_image_path = "" 
# -----------------------------------------------------------


# STEP 1
# -----------------------------------------------------------
# Step 1, use FreeSurfer to extract surface from T1 image. This is done in bash, Terminal.
# Step 1 consists of three sub-steps.
# 0) set Freesurfer subject directory environment, SUBJECTS_DIR
# export SUBJECTS_DIR=/mnt/process/Miaoc/Data/EEG-MEG/Freesurfer/
# ----SIMPLE WAY TO DO (choose either way)
# 1) add subject to workflow
# recon-all -i T1.nii -subjid P0XX_XX -all
# 2) recon this subject in workflow, please remember to put '-all' in parameter list 
# recon-all -i T1.nii -subjid P0XX_XX -3T -all -openmp 4
# recon-all -i T1.nii -subjid P0XX_XX -3T -all -openmp 12
# Or we want to run Freesurfer recon-all only upto reconstructing surfaces
# recon-all -i T1.nii -subjid P0XX_XX -openmp 8 -autorecon1 -autorecon2
# ----ROBUST WAY TO DO (choose either way)
# 1) If patient has one or multiple MRPage scans and a FLAIR scan
# recon-all -i MPR1 -i MPR2  -FLAIR FLAIR1 -subjid P0XX_XX -3T -all -openmp 4 -FLAIRpial
# 3) after freesurfer successfully finishes, run generate high-res head surface
# mkheadsurf -s PXXX_XX -srcvol T1.mgz -thresh1 30
# here, in mkheadsurf you may encounter subjdir is undefined.
# To fix this issue, add set subjdir = $SUBJECTS_DIR/$SUBJID
# -----------------------------------------------------------


# STEP 2
# -----------------------------------------------------------
# Step 2, create BEM meshes, i.e. fif file from FreeSurfer surface and create BEM model as well.
# Step 2 requires surface binary files from Step 1. Therefore make sure always do Step 1 in terminal first.
# Step 2 is done in Python environment, with MNE-Python tool imported beforehand.

# And, because we didn't implement FLASH 5/30 sequences at St Vincent's
# Hospital's radiology department. With normal T1-MPRage and T2-FLAIR scans, we use
# Freesurfer Watershed algorithm to generate BEM surfaces.

switch_generate_tri_mesh = True
if (
    BEM_method == "watershed" and switch_generate_tri_mesh == True
):  # or if only MPRage sequence
    mne.bem.make_watershed_bem(
        subject=patient_ID,
        subjects_dir=freesurfer_subjects_dir,
        overwrite=True,
        volume="T1",
        atlas=True,
        gcaatlas=False,
        preflood=1.75,
        show=True,
        verbose=True,
    )
elif BEM_method == "flash" and switch_generate_tri_mesh == True:
    mne.bem.make_flash_bem(
        subject=patient_ID,
        overwrite=True,
        show=True,
        subjects_dir=freesurfer_subjects_dir,
        flash_path=meFLASH_image_path,
        verbose=True,
    )
# elif BEM_method == 'FT' and switch_generate_tri_mesh == True:

if (
    BEM_method != "watershed"
    and BEM_method != "flash"
    and BEM_method != "curry"
    and switch_generate_tri_mesh == True
):
    bem_spacing = None

BEM_folder_path = freesurfer_subjects_dir + "/" + patient_ID + "/bem/"
# if BEM_method == "watershed" or BEM_method == "curry" or BEM_method == "ft":
#     subprocess.call(
#         [
#             "rm",
#             "-r",
#             BEM_folder_path + "inner_skull.surf",
#             BEM_folder_path + "outer_skull.surf",
#             BEM_folder_path + "outer_skin.surf",
#             BEM_folder_path + "brain.surf",
#         ]
#     )
#     # make symbol links
#     subprocess.call(
#         [
#             "ln",
#             "-s",
#             BEM_folder_path + BEM_method + "/lh.inner_skull_surface",
#             BEM_folder_path + "inner_skull.surf",
#         ]
#     )
#     subprocess.call(
#         [
#             "ln",
#             "-s",
#             BEM_folder_path + BEM_method + "/lh.outer_skull_surface",
#             BEM_folder_path + "outer_skull.surf",
#         ]
#     )
#     subprocess.call(
#         [
#             "ln",
#             "-s",
#             BEM_folder_path + BEM_method + "/lh.outer_skin_surface",
#             BEM_folder_path + "outer_skin.surf",
#         ]
#     )
#     subprocess.call(
#         [
#             "ln",
#             "-s",
#             BEM_folder_path + BEM_method + "/lh.brain_surface",
#             BEM_folder_path + "brain.surf",
#         ]
#     )
# elif BEM_method == "flash":
#     subprocess.call(
#         [
#             "rm",
#             "-r",
#             BEM_folder_path + "inner_skull.surf",
#             BEM_folder_path + "outer_skull.surf",
#             BEM_folder_path + "outer_skin.surf",
#         ]
#     )
#     # make symbol links
#     subprocess.call(
#         [
#             "ln",
#             "-s",
#             BEM_folder_path + BEM_method + "/lh.inner_skull_surface",
#             BEM_folder_path + "inner_skull.surf",
#         ]
#     )
#     subprocess.call(
#         [
#             "ln",
#             "-s",
#             BEM_folder_path + BEM_method + "/lh.outer_skull_surface",
#             BEM_folder_path + "outer_skull.surf",
#         ]
#     )
#     subprocess.call(
#         [
#             "ln",
#             "-s",
#             BEM_folder_path + BEM_method + "/lh.outer_skin_surface",
#             BEM_folder_path + "outer_skin.surf",
#         ]
#     )

# make bem model and save to subject_dir's bem folder
# Because setting a single conductivity value, it's producing
# a single layer of surface
bem_surfaces = mne.make_bem_model(
    subject=patient_ID,
    ico=bem_spacing,
    conductivity=BEM_conductivity,
    subjects_dir=freesurfer_subjects_dir,
    verbose=True,
)

fname_bem = None
if len(BEM_conductivity) == 1:
    fname_bem = (
        freesurfer_subjects_dir
        + "/"
        + patient_ID
        + "/bem/"
        + patient_ID
        + "_"
        + BEM_method
        + "_ico"
        + str(bem_spacing)
        + "_bem.fif"
    )
elif len(BEM_conductivity) == 3:
    fname_bem = (
        freesurfer_subjects_dir
        + "/"
        + patient_ID
        + "/bem/"
        + patient_ID
        + "_"
        + BEM_method
        + "_ico"
        + str(bem_spacing)
        + "_3Layers_bem.fif"
    )
# save bem surfaces to a fif file
mne.write_bem_surfaces(fname_bem, bem_surfaces, overwrite=True)

mri_slices = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 
              80, 85, 90, 95, 100, 105, 110, 115,
              120, 125, 130, 135, 140, 145, 150, 155, 
              160, 165, 170, 175, 180, 185, 190, 195, 200]
fig = mne.viz.plot_bem(
                    subject=patient_ID,
                    subjects_dir=freesurfer_subjects_dir,
                    orientation='coronal',
                    slices=mri_slices,
                    brain_surfaces='pial',
                    src=None,
                    show=True,
                    show_indices=True,
                    )
# plot_mri_bem_fig_fname = "/Users/miaoc/Documents/Clinical Neurosciences/Data/PhD Thesis/Connectivity_Simulations/Figures/Chapter2/BEM surfaces/" + \
#     patient_ID + "_" + BEM_method + "_3-layer-BEM_surf-contour-mri_" + \
#     str(len(mri_slices)) + "_slices.png"
bem_fig_fname = freesurfer_subjects_dir + '/' + patient_ID + '/bem/' + patient_ID + '_' + BEM_method + '_bem_geometry_check.png'
plt.savefig(fname=bem_fig_fname, facecolor='k')
# plt.close()

# mne.viz.plot_alignment(subject=patient_ID,
#                        subjects_dir=freesurfer_subjects_dir,
#                        surfaces=['outer_skull', 'inner_skull', 'pial'])

# # make BEM solution
bem_sol = mne.make_bem_solution(surfs=bem_surfaces, verbose=True)
bem_sol_fname = None

if len(BEM_conductivity) == 1:
    bem_sol_fname = (
        freesurfer_subjects_dir
        + "/"
        + patient_ID
        + "/bem/"
        + patient_ID
        + "_"
        + BEM_method
        + "_ico"
        + str(bem_spacing)
        + "_bem-sol.fif"
    )
elif len(BEM_conductivity) == 3:
    bem_sol_fname = (
        freesurfer_subjects_dir
        + "/"
        + patient_ID
        + "/bem/"
        + patient_ID
        + "_"
        + BEM_method
        + "_ico"
        + str(bem_spacing)
        + "_3Layers_bem-sol.fif"
    )
# save BEM solution to a fif file
mne.write_bem_solution(fname=bem_sol_fname, bem=bem_sol, overwrite=True, verbose=True)

# STEP 2.LAST
# Generate high-res skin surface in freesurfer
# freesurfer command, be cautious with freesurfer 7.2.0
# mkheadsurf -s MEGXXXX -srcvol T1.mgz -thresh1 30
# -----------------------------------------------------------
# subprocess.call(
#     [
#         "mkheadsurf",
#         "-s",
#         patient_ID,
#         "-sd",
#         freesurfer_subjects_dir,
#         "-srcvol",
#         "T1.mgz",
#         "-thresh1",
#         "30"
#     ]
# )

# # STEP 3
# # -----------------------------------------------------------
# # Step 3, set up source space
# # surface based source space
# # define the surface we use to generate sources
# surface_name = 'pial'
# if len(BEM_conductivity) == 1:
#     surf_src_fname = (
#         freesurfer_subjects_dir
#         + "/"
#         + patient_ID
#         + "/bem/"
#         + patient_ID
#         + "_"
#         + BEM_method
#         + "_ico"
#         + str(bem_spacing)
#         + "_bem_"
#         + surface_name
#         + "_surf-src.fif"
#     )
# elif len(BEM_conductivity) == 3:
#     surf_src_fname = (
#         freesurfer_subjects_dir
#         + "/"
#         + patient_ID
#         + "/bem/"
#         + patient_ID
#         + "_"
#         + BEM_method
#         + "_ico"
#         + str(bem_spacing)
#         + "_3Layers_bem_"
#         + surface_name
#         + "_surf-src.fif"
#     )

# src = mne.setup_source_space(subject=patient_ID, spacing='ico5', surface=surface_name,
#                              subjects_dir=freesurfer_subjects_dir, add_dist=False, n_jobs=4,
#                              verbose=True)
# mne.write_source_spaces(surf_src_fname, src, overwrite=True)
#
# # volumne based source space
# vol_src_spacing = 5
# if len(BEM_conductivity) == 1:
#     vol_src_fname = (
#         freesurfer_subjects_dir
#         + "/"
#         + patient_ID
#         + "/bem/"
#         + patient_ID
#         + "_"
#         + BEM_method
#         + "_ico"
#         + str(bem_spacing)
#         + "_bem_"
#         + str(vol_src_spacing)
#         + "_vol-src.fif"
#     )
# elif len(BEM_conductivity) == 3:
#     vol_src_fname = (
#         freesurfer_subjects_dir
#         + "/"
#         + patient_ID
#         + "/bem/"
#         + patient_ID
#         + "_"
#         + BEM_method
#         + "_ico"
#         + str(bem_spacing)
#         + "_3Layers_bem_"
#         + str(vol_src_spacing)
#         + "_vol-src.fif"
#     )
# vol_src = mne.setup_volume_source_space(subject=patient_ID, pos=vol_src_spacing, mri=None, bem=fname_bem,
#                                         surface=None, mindist=0.0, exclude=0.0, subjects_dir=freesurfer_subjects_dir, volume_label=None, add_interpolator=False, verbose=True)
# mne.write_source_spaces(vol_src_fname, vol_src, overwrite=True)
# # save source spacing to fif file
#
# # STEP 3-extra
# # Visualise source space with surfaces and MR scan.
# mne.viz.plot_bem(subject=patient_ID, subjects_dir=freesurfer_subjects_dir,
#                  brain_surfaces='pial', src=src, orientation='coronal')
# -----------------------------------------------------------


# STEP 4
# -----------------------------------------------------------
# Step 4, make forward model
# fname_trans = freesurfer_subjects_dir + '/' + \
#     patient_ID + '/bem/' + patient_ID + '-trans.fif'
#
# fname_meg_raw = '/mnt/cifs/emeg/EEG-MEG/P037_AB/MEG/P037_AB_eyesclosed_tsss.fif'
#
# fwd = mne.make_forward_solution(fname_meg_raw, trans=fname_trans, src=src, bem=bem_sol,
#                                 meg=True, eeg=False, mindist=5.0, n_jobs=2)
# -----------------------------------------------------------
