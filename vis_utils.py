"""Set of commonly used utility functions"""


import glob
import os
import shutil

import numpy


def get_data_paths(data_dir, pol, in_days, in_times=None, file_format=None):
    """Retrieve the paths of selected visibility datasets

    :param data_dir: Directory where miriad datasets live
    :type data_dir: str
    :param in_days: JDs of visibilities to select
    :type in_days: list
    :param in_times: Decimal part of JD to select
    :type in_times: list
    :param file_format: Format of data to be selected (e.g. miriad -> uv)
    :type file_format: str

    :return: Paths of datasets
    :rtype: list
    """
    if not file_format:
        file_format = '*'
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


def flt_data_paths(in_data):
    """Removes datasets that have already been processed

    :param in_data: Paths of datasets to process
    :type in_data: list

    :return: Filtered paths of datasets to process
    :rtype: list
    """
    in_data_filtered = list(in_data) # copying list of all sessions
    for data_path in in_data:
        if os.path.exists(os.path.join(procdir, os.path.split(data_path)[-1][:-len('uv')]+'ms.npz')):
            in_data_filtered.remove(data_path)
    return in_data_filtered


def cleanspace(dir):
    """Removes all files in specified directory"""
    shutil.rmtree(dir, ignore_errors=True)
    os.mkdir(dir)


def find_nearest(arr, val):
    """Find nearest value in array and its index

    :param array: Array-like
    :type array: array-like
    :param val: Find nearest value to this value
    :type val: float, int

    :return: Tuple of nearest value to val in array and its index
    :rtype: tuple
    """
    arr = numpy.asarray(arr)
    idx = (numpy.abs(arr - val)).argmin()
    return arr[idx], idx
