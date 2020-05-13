"""Antenna and baseline information for the H1C_IDR2 HERA dataset"""


import numpy as np


bad_bls = [[50, 51]] # [66, 67], [67, 68], [68, 69],[82, 83], [83, 84], [122, 123]]
hera_resolution = 97656.25 # MHz
hera_chans = np.arange(1024, dtype=int)

# IDR2 dataset
idr2_jds = [2458098, 2458099, 2458101, 2458102, 2458103, 2458104, 2458105, \
            2458106, 2458107, 2458108, 2458109, 2458110, 2458111, 2458112, \
            2458113, 2458114, 2458115, 2458116, 2458140]

# Antennas removed as data from these shown to be bad: 86, 88, 137, 139
idr2_ants = [0, 1, 2, 11, 12, 13, 14, 23, 24, 25, 26, 27, 36, 37, 38, 39, 40, \
             41, 50, 51, 52, 53, 54, 55, 65, 66, 67, 68, 69, 70, 71, 82, 83, \
             84, 85, 87, 120, 121, 122, 123, 124, 140, 141, 142, 143]

# Bad baselines removed: [85,86], [86, 87], [87, 88], [136, 137], [137, 138],
#                        [138, 139], [139, 140]
idr2_bls = [[0, 1], [1, 2], [11, 12], [12, 13], [13, 14], [23, 24], [24, 25], \
            [25, 26], [26, 27], [36, 37], [37, 38], [38, 39], [39, 40], \
            [40, 41], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], \
            [65, 66], [66, 67], [67, 68], [68, 69], [69, 70], [70, 71], \
            [82, 83], [83, 84], [84, 85], [120, 121], [121, 122], \
            [122, 123], [123, 124], [140, 141], [141, 142], [142, 143]]

# Map JD to known bad antennas
# Bad antennas found here: http://hera.pbworks.com/w/page/123874272/H1C_IDR2
idr2_bad_ants = np.empty((2, 19), list)
idr2_bad_ants[0, :] = idr2_jds
bad_ants_list = [
    [0, 136, 50, 2],
    [0, 50],
    [0, 50],
    [0, 50, 98],
    [0, 136, 50, 98],
    [50, 2],
    [0, 136, 50, 98],
    [0, 136, 50],
    [0, 136, 50, 98],
    [0, 136, 50],
    [137, 50, 2],
    [0, 136, 50],
    [0, 136, 50],
    [0, 50],
    [0, 136, 50, 98],
    [0, 136, 50, 11],
    [0, 136, 50],
    [0, 50, 98],
    [104, 50, 68, 117]
    ]
idr2_bad_ants[1, :] = np.array([np.array(bad_ants_jd)
                           for bad_ants_jd in bad_ants_list])
# Converting antenna numbers from HERA to CASA numbering by adding 1
idr2_bad_ants_casa = np.copy(idr2_bad_ants)
idr2_bad_ants_casa[1, :] += 1
