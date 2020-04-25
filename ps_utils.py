"""Power spectrum computation helper functions"""


from heapq import nsmallest

import numpy as np
from scipy.stats import median_absolute_deviation as mad

from vis_utils import find_nearest


# For HERA H1C_IDR2
bad_bls = [[50, 51]] # [66, 67], [67, 68], [68, 69],[82, 83], [83, 84], [122, 123]]
hera_resolution = 97656.25 # MHz
hera_chans = np.arange(1024, dtype=int)


def mod_zscore(arr):
    """Modified z-score, as defined by Iglewicz and Hoaglin

    :param arr: Array
    :type arr: array-like

    :return: Modified z-scores of the elements of the input array
    :type: ndaray
    """
    return 0.6745*(np.asarray(arr) - np.median(arr))/mad(arr)


def find_mis_days(last, jd_days, mod_z_tresh=3.5):
    """Find misaligned days by computing the modified z-scores of the first and
    last LAST timestamps, and discarding any days that have a modified z-score
    that exceeds the treshold.

    :param last: LAST
    :type last: float
    :param mod_z_tresh: Threshold of modified z-score to discard a day of data
    :type mod_z_thresh: float

    :return: Misaligned days
    :rtype: ndarray
    """
    start_last, end_last = zip(*[(last[0, i], last[-1, i]) for i in \
                                 range(last.shape[1])])
    start_zscores = np.where(mod_zscore(start_last) > mod_z_tresh)[0]
    end_zscores = np.where(mod_zscore(start_last) > mod_z_tresh)[0]
    mis_days_idx = list(set(start_zscores & end_zscores))
    mis_days = np.asarray(jd_days)[mis_days_idx]
    if mis_days:
        print('Misaligned days: {} - check alignment'.format(mis_days))
    return mis_days


class IndexVis:
    """Indexing in all dimensions

    The visibility dataset is indexed according to the flagging from:
        - misaligned days
        - known bad baselines
        - selected LASTs
        - selected baelines

    :param vis_data: Visibility dataset
    :type vis_data: NpzFile
    :param InJDs: Selected Julian days
    :type InJDs: list
    :param LAST: Selected LAST
    :type LAST: float
    :param last_tints: Selected number of time intgrations to average over
    :type last_tints: int
    :param chan_bounds: Selected upper and lower channel bounds
    :type chan_bounds: tuple
    """
    def __init__(self, vis_data, InJDs, LAST, last_tints, chan_bounds):
        self.vis = vis_data['visibilities'] # shape: (last bins, days, bls, chans)
        self.bls = vis_data['baselines'] # shape: (bl, [ant1, ant2])
        self.last = vis_data['last'] # shape: (last bins, days)
        self.days = vis_data['days'].astype(int) # JD days
        self.flags = vis_data['flags'].astype(bool) # same shape as vis
        self.InJDs = InJDs
        self.LAST = LAST
        self.last_tints = last_tints
        self.chan_bounds = chan_bounds
        self.chans = hera_chans

    def check_last(self):
        """Ensures that the selected LAST exists in the visibility dataset

        :return: Verified LAST
        :rtype: float
        """
        if self.LAST < np.min(self.last) or self.LAST > np.max(self.last):
            raise ValueError('Specify LAST value in between {} and {}'.format(
                np.ceil(np.min(self.last)*100)/100, np.floor(np.max(self.last)*100)/100))
        else:
            return self.LAST

    def check_chans(self):
        """Ensures that the selected frequency channels exist in the visibility
        dataset

        :return: Verified channel bounds
        :rtype: tuple
        """
        chan_start, chan_end = self.chan_bounds
        if chan_start < self.chans[0] or chan_end > self.chans[-1]:
            raise ValueError('Specify channels in between {} and {}'.format(\
                self.chans[0], self.chans[-1]))
        elif chan_start > chan_end:
            raise ValueError('Ensure that chan_start > chan_end')
        else:
            return (chan_start, chan_end)

    def flt_days(self):
        """Removes days that do not appear in the selected JDS or that are
        misaligned

        :return: Indices of days to keep
        :rtype: list
        """
        # Find misaligned days
        misaligned_days = find_mis_days(self.last, self.days)
        # Get the filtered list of good JDS
        flt_days = [day for day in list(set(self.InJDs) & set(self.days)) if \
                    day not in misaligned_days]
        # Get the filtered list of good JD indices
        flt_days_idx = [np.where(self.days == flt_day)[0][0] for flt_day in flt_days]
        return flt_days_idx

    def flt_bls(self):
        """Removes bad baselines

        :return: Indices of baselines to keep
        :rtype: list
        """
        # Get indices of bad baselines
        bad_bls_idxs = [np.where(self.bls == bad_bl) for bad_bl in bad_bls \
                        if bad_bl in self.bls]
        # Get the rows of the bad baselines
        bad_bls_idxs = [bad_bls_idx[0][0] for bad_bls_idx in bad_bls_idxs if \
                    bad_bls_idx[0].size==2 and len(set(bad_bls_idxs[0][0]))==1]
        flt_bls_idx = [bl_idx for bl_idx in range(self.bls.shape[0]) \
                       if bl_idx not in bad_bls_idxs]
        return flt_bls_idx

    def flt_lasts(self):
        """Selects LASTs

        :return: Indices of LASTs to keep
        :rtype: list
        """
        LAST = self.check_last()
        last_idx = find_nearest(np.median(self.last, axis=1), LAST)[1]
        flt_last_idx = sorted(nsmallest(self.last_tints, np.arange(self.last.shape[0]), \
                                         key=lambda x: np.abs(x - last_idx)))
        return flt_last_idx

    def flt_chans(self):
        """Selects frequency channels

        :return: Selected channels
        :rtype: ndarray
        """
        chan_bounds = self.check_chans()
        chans_range = np.arange(self.chan_bounds[0]-1, self.chan_bounds[1])
        return chans_range

    def do_indexing(self):
        """Only selecting good days and good baselines, and only choosin
        specified channels and LASTs

        :return: Filtered components of visibility dataset
        :rtype: tuple
        """
        last_indexing = self.flt_lasts()
        day_indexing = self.flt_days()
        bl_indexing = self.flt_bls()
        chan_indexing = self.flt_chans()
        vis_indexing = np.ix_(last_indexing, day_indexing, bl_indexing, \
                              chan_indexing)
        vis = self.vis[vis_indexing]
        bls = self.bls[bl_indexing, :]
        last = self.last[np.ix_(last_indexing, day_indexing)]
        days = self.days[day_indexing]
        flags = self.flags[vis_indexing]
        chans = hera_chans[chan_indexing]
        return vis, bls, last, days, flags, chans
