"""Set of commonly used utility functions"""


import glob
import os
import shutil


def get_data_paths(data_dir, pol, in_days, in_times=None, file_format='uv'):
    """Retrieve the paths of selected visibility datasets

    Select the datasets to by JD and LST
    :param data_dir: Directory where miriad datasets live
    :param in_days: JDs of visibilities to select
    :param in_times: Decimal part of JD to select
    :param file_format: Format of data to be selected (e.g. miriad -> uv)
    """
    in_data = []
    for in_day in InDays:
        if in_times:
            for in_time in in_times:
                [InData.append(g) for g in glob.glob(
                    os.path.join(data_dir, str(in_day), pol,
                        "*.{}*.{}".format(in_time, file_format)))]
        else:
            [InData.append(g) for g in glob.glob(
                os.path.join(data_dir, str(in_day), pol,
                    '*.{}'.format(file_format)))]
    return sorted(in_data)


def flt_dat_paths(in_data):
    """Removing datasets that have already been converted"""
    in_data_filtered = list(in_data) # copying list of all sessions
    for data_path in in_data:
        if os.path.exists(os.path.join(procdir, os.path.split(data_path)[-1][:-len('uv')]+'ms.npz')):
            in_data_filtered.remove(data_path)
    return in_data_filtered


def cleanspace(dir):
    """Removes all files in specified directory"""
    shutil.rmtree(dir, ignore_errors=True)
    os.mkdir(dir)
