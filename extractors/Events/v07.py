"""
Event parser for pSWM touchscreen setup
"""

import sys
import pandas as pd
import numpy as np
import pingparser.general as genparse
import pyfun.bamboo as boo
import pyfun.timestrings as tstr
import pingparser.runningval as runningval

#three cases:
#10.16 -- is_warmup variable introduced (but unused). Can ignore
#10.22 -- warmup_state: integer which counts number of warmup processes occurring for a given trial.
#         0 = not warm up trial.
#         Additionally, trial params pushing has been cleaned up so that key trial params are only pushed at trial start
#         through event ping. Running valuators become unnecessary (just grab the value within trial). However, we keep
#         them for backward compatibility in this version of event extractor.

#10/25, 28, 29:
#push trial params never completes  because optoTrial not received, leading to multiple instances of same variable pushed
#(number of instances increments by one per trial i.e. on trial 2, 2 pushes, trial 100, 100 pushes :(



class Extractor:
    VERSION = "touch_v07"
    DATE = "10.31.24"
    TYPE = 'Events'

    COLNAMES = [
        'TrialNum', 'FixationDur', 'RespError_cuefrac', 'Cue_D', 'CueMove', 'RespPause',
        'RespBox_type', 'CueRel1_x', 'CueRel1_y', 'CueRel2_x', 'CueRel2_y', 'Cue1_x', 'Cue1_y',
        'Cue2_x', 'Cue2_y', 'anchor1_x', 'anchor1_y', 'anchor2_x', 'anchor2_y', 'ITI', 'Welzl_x',
        'Welzl_y', 'Welzl_D', 'WMDelay', 'WMTrial', 'optoTrial', 'TrialType', 'TrialDur',
        'FixationBreaks', 'warmup_state', 'FixationGraceDur'
    ]

    RUNNING_VALS = ['Cue_D', 'FixationDur', 'TrialType', 'RespPause', 'WMDelay']


    def __init__(self):
        self.RAW_TO_DELETE = []  # list of dicts for slicing away raw df in pre_processor()
        self.row_holder_template = pd.DataFrame({'val': None}, index=self.COLNAMES)


    def extract(self, fn, sess_name):
        """
    	Process one session raw file and return one row per trial,
    	with columns containing per-trial variables (e.g., stimulus, animal choices).
    	"""

        df_sess_raw = genparse.read_raw(fn)  # Load raw data

        if len(df_sess_raw) == 0:
            return pd.DataFrame()

        # Pre-process the raw session data
        df_sess_raw = self._pre_processor(df_sess_raw)

        # === Extract values trial by trial and compile into a list ===
        sess_extracts = []
        for TrialNum, df_trial in df_sess_raw.groupby('TrialNum'):
            one_trial = self._trial_summarizer(df_trial)
            if one_trial is not None:
                sess_extracts.append(one_trial['val'].tolist())

        # Convert list of extracted trials into a DataFrame
        df_sess = pd.DataFrame(sess_extracts, columns=self.COLNAMES)

        if df_sess.empty:
            return pd.DataFrame()

        # === Compute running values for the entire session ===
        running_vals = self._running_valuator(df_sess_raw)
        for subj_name, vals in running_vals.items():
            try:
                df_sess = boo.merge_update(df_sess, vals, subj_name, match_column='TrialNum')
            except KeyError:
                continue  # Skip if 'TrialNum' does not exist due to empty DataFrame

        # Add session identifier
        df_sess['sess'] = sess_name

        # Post-process the session data
        df_sess = self._post_processor(df_sess)

        return df_sess

    def _calc_trial_dur(self, df_trial, start_event='ModEvent', end_event='Trial_ended'):
        """Calculate start and end times of trial."""
        start_end = []
        for event in [start_event, end_event]:
            try:
                # take the first ModEvent and first Trial_ended (but there should only be one!)
                timestr = boo.slice(df_trial, {'Subject': [event]})['Timestamp'].iloc[0]

                start_end.append(pd.to_datetime(timestr))
            except:
                start_end.append(None)

        trial_dur = tstr.calc_dt(start_end[1], start_end[0], 'secs')
        return trial_dur

    def _pre_processor(self, df_sess_raw):
        """Pre-process raw session data based on configured RAW_TO_DELETE."""
        df = df_sess_raw.copy()
        for r in self.RAW_TO_DELETE:
            df = boo.slice(df, r, '-')
        return df

    def _post_processor(self, df):

        #1. create variable for CueMove
        df['CueMove'] = (df['Cue1_x'] != df['Cue2_x']) | (df['Cue1_y'] != df['Cue2_y'])
        #2. convert WMTrial variable from string to boolean
        df['WMTrial'] = df['WMTrial'] == 'True'

        #3. typecast
        for subj in ['WMDelay', 'Cue_D']:
            df[subj] = df[subj].astype('float')

        #4. remove trial 0
        df = boo.slice(df, {'TrialNum': [0]}, '-')

        #5. set tiny values of WMDelay to be 0
        # the cause of this was related to C# and floating point values in the script WM_FixationWarmUp
        # (in use around Oct 16-22)
        epsilon = 1e-10
        df['WMDelay'] = df['WMDelay'].apply(lambda x: 0 if abs(x) <= epsilon else x)

        #set non-WM trials types WMDelay to nan
        nonWMs = boo.slice(df, {'TrialType': ['1_PortFix', '2_PortFixCue', '3_WMDist']})
        df.loc[nonWMs.index, 'WMDelay'] = np.nan

        return df

    def _running_valuator(self, df_sess_raw):
        """Extract running values."""
        return runningval.get(df_sess_raw, self.RUNNING_VALS, debug=False)

    def _trial_summarizer(self, df_trial):
        """
    	df_trial: df containing all events restricted to a single trial.
    	return crucial trial info as a single row
    	crucial trial info differs across experiments
    	"""

        row_holder = self.row_holder_template.copy() #.copy() here is deep copy by default
        row_holder.loc['TrialNum', 'val'] = df_trial['TrialNum'].iloc[0]

        # calculate trial duration
        trial_dur = self._calc_trial_dur(df_trial)
        row_holder.loc['TrialDur', 'val'] = trial_dur

        # count number of fixation breakings
        n_fix_breaks = len(boo.slice(df_trial, {'Value': ['FixationInterrupted']}))
        row_holder.loc['FixationBreaks', 'val'] = n_fix_breaks

        # find fixation completed event
        fixated = boo.slice(df_trial, {'Value': ['FixationCompleted']}, '+') # index where InTimerCompleted
        if len(fixated) == 0:
            #no fixation completed
            fixated_row = df_trial.index[-1] #just set to last row in trial
        else:
            # save row number, for finding parameters centered around fixation complete
            # e.g. cue position, while discarding stimulus parameters occurring for broken fixations
            fixated_row = fixated.index[0]

        # === Extract trial individual params ===
        for param in ['ITI', 'RespError_cuefrac']:
            row_holder.loc[param, 'val'] = genparse.get_trial_param(df_trial, param, dtype='float', single='strict')
        for param in ['Welzl_D', 'FixationGraceDur']:
            row_holder.loc[param, 'val'] = genparse.get_trial_param(df_trial, param, dtype='float', single='last')
        for param in ['warmup_state']:
            row_holder.loc[param, 'val'] = genparse.get_trial_param(df_trial, param, dtype='int', single='last')

        # Extract xy subjects that are not prepost (below)
        for xy_param in ['Welzl']:
            xy_str = genparse.get_trial_param(df_trial, xy_param + '_xy', dtype='string', single='last')
            xy = genparse.str_to_list(xy_str) if xy_str else (None, None)
            row_holder.loc[xy_param + '_x'], row_holder.loc[xy_param + '_y'] = xy

        for param in ['RespBox_type', 'WMTrial', 'optoTrial']:
            row_holder.loc[param, 'val'] = genparse.get_trial_param(df_trial, param, dtype='string', single='last')

        for param, prefix in zip(['Cue_xy', 'anchor_xy', 'Cue_xyRel'], ['Cue', 'anchor', 'CueRel']):
            param_prepost = genparse.closest_to_row(df_trial, param, fixated_row)

            # TODO: can improve for efficiency
            for i, v in enumerate(param_prepost):
                if v is None:
                    param_prepost[i] = '(nan,nan)'

            output = genparse.format_prepost_xy(param_prepost, prefix)
            for k, v in output.items():
                row_holder.loc[k, 'val'] = v

        return row_holder

# Example usage for testing when running the module directly
if __name__ == "__main__":
    import sys
    import os

    # Example file path and session name for testing
    fn = "Y:/Edmund/Data/Touchscreen_pSWM/raw/EC05/2024_10_20-10_40/results_2024-10-20T10_40_14/Events.csv"
    subsess_name = '2024-10-20'

    # Check if the file exists
    if not os.path.exists(fn):
        print(f"Test file not found: {fn}")
        sys.exit(1)

    # Create an instance of the Extractor
    extractor = Extractor()

    # Run the extraction process on the test data
    df_sess = extractor.extract(fn, subsess_name)

    # Print or display the output for testing
    print("Extracted Session Data:")
    print(df_sess)