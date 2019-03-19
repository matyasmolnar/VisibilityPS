### HERA LST Aligner
#
# James Kent, University of Cambridge
# jck42@cam.ac.uk
#
# modified by Matyas Molnar
#
# This script takes a directory of visibility files outputted from recipe using
# heracasa (Bojan Nikolic, bn204@cam.ac.uk), and from there bins them so that
# visibilities over successive days are aligned in LST.
#
# scp /Users/matyasmolnar/HERA_Data/VisibilityPS/align_lst.py mdm49@login-cpu.hpc.cam.ac.uk:/rds/project/bn204/rds-bn204-asterics/mdm49
#
# running script
# casapython align_lst.py --lst_start 3.05 --lst_end 3.71 --date_start 2458098 --date_end 2458140 --output_file 'aligned_raw_visibilities' /rds/project/bn204/rds-bn204-asterics/mdm49/IDR2_raw/ /rds/project/bn204/rds-bn204-asterics/mdm49/aligned/


from __future__ import print_function

from astropy.time import Time
import matplotlib.pyplot as plt
import os
import sys
import numpy
import argparse
import textwrap



class LST_Binner(object):

    """
    Class that takes the aligned LSTs and a dictionary of ALL the visibilities from
    all input days and concatenates them together and outputs them as a
    single numpy array.

    Attributes:

    lst_array     [Numpy array] This is an array of all aligned LSTS. We take the columns
                  of this to index {visibility_dict} to get the visibilities for the
                  correct LST.

    visibility_dict  [Dictionary] Two-layer dictionary, keyed by date then by LST.
                  Contains the visibilities as a [Numpy array] of shape
                  (no_baselines,no_channels).

    lst_start     [Float] Start of LSTs, in fractional hours.

    lst_end       [Float] End of LSTs, in fractional hours.

    # lst_range     [Integer] Range of LST's to bin.

    baseline_no      [Integer] Number of baselines in visibilities in [visibility_dict].

    channels      [Integer] Number of channels in visibilities in [visibility_dict].

    day_array     [Numpy array] of shape (no_days). Gives UTC Dates.

    outlst_array  [Numpy array] of shape (no_lsts,no_days)Exact LST's of visibilities outputted.

    data_array    [Numpy array] of shape (no_lsts,no_days,no_baselines,no_channels). Contains
                  visibilities.

    flag_array    [Numpy array] of shape (no_lsts,no_days,no_baselines,no_channels). Contains
                  individual channel flags from the real time processor(RTP).

    baseline_array   [Numpy array] of shape (no_baselines,2). Contains antennas which make up the
                  individual baselines.

    Member Functions:

    __init__() Initialises an instance of class LST_Binner

    __bin_lsts() Where the magic happens. Takes our aligned LST's and outputs the visibilities.

    __save_binned_lsts() Saves day_array, outlst_array and data_array to a .npz file.


    TODO: Work out what to do with the bits on the end we don't care about.
    TODO: Averaging
    """

    def __init__(self,
                 lst_array,
                 visibility_dict,
                 lst_start,
                 lst_end,
                 baseline_no = 35,
                 channels = 1024):
        """
        Instantiates the LST_Binner class which takes a numpy array of aligned LST's and
        a dictionary describing all of our visibility data and outputs the items of interest.

        Inputs:

        lst_array     [Numpy Array] Array of aligned LST's. Individual elements used as
                      keys for visibility_dict.

        visibility_dict  [Dictionary] Three-layer dictionary with all the visibility information,
                      as well as flags, baselines, days etc.

        lst_start     [Integer] Column index for [lst_array] to output binned LST's from.

        lst_range     [Integer] Range of LST's to bin.

        baseline_no      [Integer] Number of baselines in dataset.

        channels      [Integer] Number of frequency channels.

        """

        #Parameters
        self.lst_array = lst_array
        self.visibility_dict = visibility_dict
        self.lst_start = lst_start
        self.lst_end = lst_end
        self.baseline_no = baseline_no
        self.channels = channels
        #Our output arrays for our LST binned data.
        self.day_array = None
        self.outlst_array = None
        self.data_array = None
        self.flag_array = None
        self.baseline_array = None
        self.sigclip_array = None

        # Calculate lst_array indices from lst_start lst_end
        lst_subset = self.lst_array[0,:]
        lst_subset = lst_subset % 1
        lst_subset = lst_subset * 24
        ind_range = numpy.argwhere((lst_subset>self.lst_start)&(lst_subset<self.lst_end))
        self.lst_start_ind = numpy.min(ind_range)
        self.lst_end_ind = numpy.max(ind_range)
        self.lst_range = ind_range.shape[0]
        self.lst_range = self.lst_end_ind - self.lst_start_ind

    def __project_unit_circle(self,visibilities):
        """
        Project angular data onto unit circle / convert to cartesian
        co-ordinates.

        Inputs:

        visibilities    [Numpy Array] Binned visibilities of shape [lst, day,
                        baseline, channel]

        """

        return numpy.cos(visibilities),numpy.sin(visibilities)

    def __calc_R(self, x, y, axis=0):
        averaged_x = numpy.mean(x,axis=axis)
        averaged_y = numpy.mean(y,axis=axis)
        return numpy.sqrt(numpy.square(averaged_x) + numpy.square(averaged_y))

# rewrite this for visibilities
    def __sigma_clip_baselines(self, visibilities, chan_threshold=100, sigma=1.0):
        """
        Takes visibilities and runs a circular sigma clip across them.
        If a particular baseline is not behaving at a particular record/day
        in > chan_threshold channels, then put a 1 in a flag array.

        Inputs:

        visibilities   [Numpy Array] Binned visibilities of shape
                       [lst, day, trid, channel]
        chan_threshold [Integer] Number of channels that a baseline
                       is off in before a flag gets set.
        sigma          [Float] Sigma value  to exceed before a baseline
                       is considered errant.

        Returns:

        fl             [Numpy Array][Bool] Flag array.

        """
        cls = visibilities.shape[:2]
        flr = numpy.zeros(shape=visibilities.shape,dtype=numpy.bool)

        for t in numpy.arange(visibilities.shape[0]):
            for d in numpy.arange(visibilities.shape[1]):
                for c in numpy.arange(visibilities.shape[3]):
                    tr = visibilities[t,d,:,c]
                    tr_x, tr_y = self.__project_unit_circle(tr)
                    averaged_x = numpy.mean(tr_x)
                    averaged_y = numpy.mean(tr_y)

                    r = numpy.sqrt(numpy.square(averaged_x) + numpy.square(averaged_y))
                    av_ang = numpy.arctan2(averaged_y,averaged_x)
                    sigma = numpy.sqrt(-2 * numpy.log(r))

                    for baseline in numpy.arange(visibilities.shape[2]):
                        if numpy.abs(visibilities[t,d,baseline,c] - av_ang) > sigma:
                            flr[t,d,baseline,c] == True

        agg = numpy.zeros(shape=cls,dtype=numpy.int32)

        for t in numpy.arange(flr.shape[0]):
            for d in numpy.arange(flr.shape[1]):
                for tr in numpy.arange(flr.shape[2]):
                    for c in numpy.arange(flr.shape[3]):
                        if flr[t,d,tr,c] == True:
                            agg[t,d,tr]+=1
        fl = agg
        return fl



    def bin_lsts(self,sigclip=False):

        """
        Takes our binned LST_Array and uses it to index our dictionary of all
        visibilities. Then extracts the LST's of interest and saves them to
        day_array, outlst_array, data_array. See class description.

        Inputs:

        sigclip   [Bool] Wether to mark out poor baselines using a sigma clip based strategy.
        """

        # Describes the days in our LST bins
        self.day_array = numpy.zeros(shape=len(self.visibility_dict.keys()))
        # Gives exact LST's for our aligned bins (for reference)
        self.outlst_array = numpy.zeros(shape=(self.lst_range, len(self.visibility_dict.keys())))
        # Final concatenated visibilities, aligned by LST
        self.data_array = numpy.zeros(shape=(self.lst_range,
                                        len(self.visibility_dict.keys()),
                                        self.baseline_no,
                                        self.channels),dtype=numpy.complex_)
        # All of the RTP flags, aligned with visibilities.
        self.flag_array = numpy.zeros(shape=(self.lst_range,
                                        len(self.visibility_dict.keys()),
                                        self.baseline_no,
                                        self.channels),
                                      dtype=numpy.int8)
        # Our baselines.
        self.baseline_array = numpy.zeros(shape=(self.baseline_no, 2))
        print("    Extracting Fields of Interest... ", end="")
        sys.stdout.flush()
        for i, date in enumerate(sorted(self.visibility_dict.keys())):
            self.day_array[i] = int(date)
            for lst in range(self.lst_range):

                lst_index = self.lst_array[i,self.lst_start_ind+lst]
                self.outlst_array[lst,i] = (lst_index %1) * 24 # Convert to fractional hours.
                visibilities_at_lst = self.visibility_dict[date][lst_index]['visibility']
                flags_at_lst = self.visibility_dict[date][lst_index]['flags']

                self.data_array[lst,i,:,:] = visibilities_at_lst
                self.flag_array[lst,i,:,:] = flags_at_lst
                self.baseline_array = self.visibility_dict[date][lst_index]['baselines']
        print("done")
        if sigclip == True:
            print("    Sigma Clipping... ", end="")
            sys.stdout.flush()
            self.sigclip_array = self.__sigma_clip_baselines(self.data_array)
            print("done")

    def save_binned_lsts(self,filename):

        """
        Takes our binned visibilities and outputs them to a .npz file.

        Inputs:

        filename    [String] Filename to save .npz file.
        """

        if self.day_array is not None:
            if (self.sigclip_array is None):
                numpy.savez(filename,
                            days=self.day_array,
                            last=self.outlst_array,
                            visibilities=self.data_array,
                            flags=self.flag_array,
                            baselines=self.baseline_array)
            else:
                numpy.savez(filename,
                            days=self.day_array,
                            last=self.outlst_array,
                            visibilities=self.data_array,
                            flags=self.flag_array,
                            baselines=self.baseline_array,
                            sigclip=self.sigclip_array)

        else:
            raise ValueError("LST's not binned")

class LST_Alignment(object):
    """
    Class takes a dictionary of datestamps and aligns them to each other.
    We take advantage of sidereal time moving 235.9 seconds per day with respect
    to the solar day.

    Attributes:

    visibility_directory      [String] Base directory where heracasa .npz files are located.

    date_set               [List] Ordered dates , earliest -> latest

    date_dict              [Dictionary] Dictionary of all dates with filenames.

    timestamp_delta        [Float] Delta between sidereal day and solar day.

    delta_ind              [Int] How many timestamp indices drift per sidereal day.

    delta_rem              [Float] Remainder from delta_ind calculation.

    Member Functions:

    __init__()             Initialises an instance of class LST_Alignment.

    __extract_visibilities()   Opens all the .npz files and concatenates them into a single dictionary
                           keyed by date/lst.

    __align_visibilities()     Aligns the visibilities by LST.

    align_timestamps()     Public function which builds the date_dict and aligns the LST's, and
                           returns them.

    TODO: Implement destructive alignment.
    """

    def __init__(self,
                 visibility_directory,
                 ordered_dates,
                 date_dict,
                 integration_time=10.7,
                 sidereal_delta=235.909,
                 destructive = True):
        """
        Initialises the LST_Alignment Class which aligns the LST's over successive
        Julian Dates"

        Inputs:

        visibility_directory   [String] Root directory of heracasa .npz files.

        ordered_dates       [List] All datestamps in order. Earliest -> Latest

        date_dict           [Dictionary] of all dates with filenames.

        integration_time    [Float] HERA Integration Time. Default = 10.7s.

        sidereal_delta      [Float] Delta between sidereal day and solar day.

        destructive         [Bool] Destructive alignment or not? [NOT IMPLEMENTED (YET)]
        """
        self.visibility_directory = visibility_directory
        self.date_set = ordered_dates
        self.date_dict = date_dict
        self.timestamp_delta = sidereal_delta / integration_time
        self.delta_ind = int(round(self.timestamp_delta)) #Get closest indice
        self.delta_rem = self.timestamp_delta % 1
        self.baseline_no = 0


    def __extract_visibilities(self):

        """
        Opens all of the .npz files in date_dict, and sorts them into a new dictionary
        which is keyed by both date and LST. Thus the key mechanism is as so:

        |
        | - Date_1 - LST_1 - 'visibilities' - [Numpy Array]
        |                  - 'flags' - [Numpy Array]
        |                  - 'baselines' - [Numpy Array]
        |	       - LST_2
        |          - LST_N
        | - Date_2 - LST_1
        |
        | - etc

        """

        visibility_dict = {}
     #Only works in Python 2.x
        for date, npz_files in sorted(self.date_dict.iteritems()):
            print(".",end="")
            sys.stdout.flush()
            visibility_dict[date] = {}
            for npz_file in npz_files:
                with numpy.load(self.visibility_directory + npz_file) as data:

                    # LST's is used to build the keys of the second tree layer,
                    # as we want to organise by LST for our alignment.
                    lsts = data['LAST']

                    visibilities = data['vis']
                    if self.baseline_no == 0:
                        self.baseline_no = visibilities.shape[0]
                    flags = data['flags']
                    baselines = data['bl']
                    for i,lst in enumerate(lsts):
                        visibility_dict[date][lst] = {}
                        visibility_dict[date][lst]['visibility'] = visibilities[:,0,:,i]
                        visibility_dict[date][lst]['flags'] = flags[:,0,:,i]
                        visibility_dict[date][lst]['baselines'] = baselines

        return(visibility_dict)

    def __align_visibilities(self, reference_lst, visibilities):

        """
        Does the alignment of the LST's over successive Julian Days.
        Little bit shakey and basic but works okay for now.

        Inputs:

        reference_lst  [Dictionary] First LST in dataset for reference.
                       Can get rid I think.

        visibilities       [Dictionary] Dictionary of all visibilities.


        TODO: Tidy this up, reference_lst not needed?
        """

        #Generate Numpy array from visibility_dictionary. Makes life significantly easier.
        lst_array = numpy.zeros(shape=(len(visibilities.keys()),len(reference_lst)))
        i = 0
        initial_date, initial_lsts = sorted(visibilities.iteritems())[0]
        for date, lst_s in sorted(visibilities.iteritems()):
            print(".",end="")
            sys.stdout.flush()
            #print(lst_s)
            #print(len(lst_s))
            for j,lst in enumerate(sorted(lst_s)):
                lst_array[i,j] = lst
            i = i + 1

        #Align LST's.

        offset_array = numpy.zeros(shape=(len(visibilities.keys())))
        datelist = reversed(self.date_set)
        prev_date = None
        for i, date in enumerate(reversed(self.date_set)):
            if i == 0:
                offset_array[i] = 0
            else:
                date_delta = int(prev_date) - int(date)
                offset_array[i] = offset_array[i-1] - (self.delta_ind - self.delta_rem)  * date_delta

            prev_date = date
        offset_array = numpy.rint(offset_array)
        offset_array = numpy.flipud(offset_array)
        offset_array = offset_array.astype(int)

        for i in range(numpy.shape(lst_array)[0]):
            lst_array[i] = numpy.roll(lst_array[i], offset_array[i]) #Bit of a hatchet job...

        # Because of the fact HERA only observes for part of the day, we end up with some records
        # eventually "drifting" out of our aligned LST window. As we roll the array to do the
        # alignment we can mask off these loose ends.
        lst_array = numpy.ma.masked_array(lst_array)
        unaligned_index = numpy.shape(lst_array)[1] + offset_array[0]
        lst_array[:,unaligned_index:] = numpy.ma.masked

        return lst_array

    def align_timestamps(self):
        """
        Generates the visibility_dictionary and generates a numpy array of aligned
        LST's, which can be used by the LST_Binner class to extract visibilities of
        choice across successive days.
        """
        print("Aggregating visibility dictionary to array (can take a while)...")
        visibility_dict = self.__extract_visibilities()
        print("done")
        #print visibility_dict
        lst_ref = self.date_set[0]
        #print(lst_ref)
        lst_ref = visibility_dict[lst_ref] #We interpolate to the LST's from the first day of observations.
        #print(lst_ref)
        # print(len(lst_ref))
        print("Performing alignment...")
        aligned_lsts = self.__align_visibilities(lst_ref, visibility_dict)
        print("done")
        return aligned_lsts, visibility_dict


#This parses all of the files and breaks them up into datestamps. Each datestamp is then aligned correctly.
class Julian_Parse(object):
    """
    Class to take a set of filepaths spat out from heracasa and create a dictionary of all the
    files, keyed by their Julian Date.

    Also returns the set of all dates found in the directory.

    Attributes:

    filepaths              [List] All filepaths from the directory ending in .npz. Assumed to be heracasa.

    file_dict              [Dictionary] Dictionary of npz files, keyed by their date.

    date_set               [Set] The set of dates.

    Member Functions:

    __init__()             Initialises an instance of class Julian_Parse.

    __find_unique_dates()  Finds all unique dates and returns the ordered list of the set of dates.

    __build_visibility_dict() Takes the ordered list of the set of dates, and filepaths and sorts
                           them into file_dict.

    break_up_datestamps()  Creates file_dict.

    return_datestamps()    Returns self.file_dict, self.date_set.

    TODO: Some sort of date ranging so you can control how much data we push through the binner.
    """



    # We assume all .npz files have 60 timestamps in them.
    def __init__(self, filepaths, date_start, date_end, npz_size = 60):

        """
        Initialises the Julian Parse Class which takes a full directory of
        heracasa .npz files and splits them by date.

        """

        self.filepaths = filepaths
        self.date_start = date_start
        self.date_end = date_end
        self.file_dict = None
        self.date_set = None

    # Parses all datestamps and converts to a set (finds unique values)
    def __find_unique_dates(self, filepaths):

        """
        Find unique dates from a directory of heracasa files.

        Inputs:

        filepaths [List] List of all .npz filepaths

        """

        detected_datestamps = []

        for file in self.filepaths:
            detected_datestamps.append(file.split('.')[1])
        detected_datestamps = set(detected_datestamps) # Convert list to set.
        detected_datestamps = sorted(detected_datestamps) # Convert set to ordered list.
        return(detected_datestamps)

    # Takes set of datestamps and filepaths and sorts them into a dict.
    # Dict layout is [Key: datestamp, Data: list of all files in that datestamp]
    def __build_visibility_dict(self, datestamps, filepaths):

        """
        Splits the filepaths into a dictionary keyed by their Julian Date

        Inputs:

        datestamps [List] An ordered list of the datestamps.

        filepaths [List] List of al .npz filepaths.

        """
        file_dict = {}

        for datestamp in datestamps:
            file_dict[datestamp] = [] #Empty list.

        for datestamp in datestamps:

            for file in filepaths:
                if file.split('.')[1] == datestamp:
                    file_dict[datestamp].append(file)
            file_dict[datestamp] = sorted(file_dict[datestamp]) #Holy shit this just works? How awesome is that.
        #print(file_dict)
        return(file_dict)



    # This builds a dictionary of all unique datestamps and their .npz files
    def break_up_datestamps(self):

        """
        Breaks up the directory of .npz files into a dictionary, keyed by date.

        """
        print("Parsing visibility directory... ", end="")
        detected_datestamps = self.__find_unique_dates(self.filepaths)
        print("done")
        print("Discovering dates within specified range...", end="")
        detected_datestamps = sorted(list(filter(lambda el: int(el) >= self.date_start and int(el) <= self.date_end, detected_datestamps)))
        print("done")
        print("Detected Dates: ")
        print(detected_datestamps)
        print("Building dictionary of dates and filepaths...", end="")
        file_dict = self.__build_visibility_dict(detected_datestamps, self.filepaths)
        print("done")
        self.file_dict = file_dict
        print(self.file_dict.keys())
        self.date_set = detected_datestamps

    # Returns dictionary.
    def return_datestamps(self):

        """
        Returns file_dict and date_set

        """
        return self.file_dict, self.date_set



def main():


    # Parse directory for input/output.
    command = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent('''
    ---------------------------------------------------------------------------------------
    HERA LST Aligner and Binner

    Author: James Kent
    Institution: University of Cambridge, 2018
    Email: jck42@cam.ac.uk

    Takes a directory of heracasa created visibilities (best created using recipe),
    and concatenates all of the .npz files together, aligns the sidereal times, and
    outputs and LST range of your choice.

    Can also optinally average visibilities across LSTs as well as calculate the standard
    deviations, which can be helpful in telling you if you there are days/times/baselines
    that are not behaving as wanted.
'''))
    command.add_argument('filepath', help = ('Directory of .npz files generated from heracasa'))
    command.add_argument('working_directory', help = ('where aligned .npz files will be put.'))
    command.add_argument('--output_file',required=False, default='output_bin',metavar='O', type=str,
                         help='Output file name.')
    command.add_argument('--sigma_clip', action='store_true', required=False,
                         help='Flags baselines based on a sigma clipping flagging strategy.')
    command.add_argument('--lst_start', required=True, metavar='S', type = float,
                         help='Start of LSTs, in fractional hours (3.2, 23.1 etc).')
    command.add_argument('--lst_end', required=True, metavar='R', type = float,
                         help='End of LSTs, in fractional hours (3.4, 23.5 etc).')
    command.add_argument('--date_start', required=True, metavar='R', type = int,
                         help='Start date for alignment.')
    command.add_argument('--date_end', required=True, metavar='R', type=int,
                         help='End date for alignment.')
    command.add_argument('--channel_number',required=False,default=1024, metavar='C', type = int,
                         help='Number of channels')
    args = command.parse_args()

    if (os.path.exists(args.working_directory) == False):
        os.makedirs(args.working_directory)

    # Find all .npz files. Assume they are all from heracasa...
    files = []
    for file in os.listdir(args.filepath):

        if file.endswith(".npz"):
            files.append(file)

    parser = Julian_Parse(files,args.date_start,args.date_end)
    parser.break_up_datestamps()
    files, dates = parser.return_datestamps()

    print("Number of days: %d"%len(dates))
    # Instantiate LST_alignment class and align timestamps.
    print("Aligning LST's (use first day as reference)...")
    aligner = LST_Alignment(args.filepath,dates,files)
    aligned_lsts, visibilities = aligner.align_timestamps()
    # Instantiate LST_binner class class, then bin LSTS and save to file.
    print("Bin LST's...")
    binner = LST_Binner(aligned_lsts, visibilities, lst_start=args.lst_start, lst_end=args.lst_end,baseline_no=aligner.baseline_no,channels=args.channel_number)
    binner.bin_lsts(sigclip=args.sigma_clip)
    print("done")
    binner.save_binned_lsts(args.working_directory+args.output_file+".npz")


if __name__=="__main__":
    main()
