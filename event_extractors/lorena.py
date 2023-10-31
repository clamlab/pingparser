"""
event parser for pSWM Lorena setup
"""

import pandas as pd
import pingparser.general as genparse
import pyfun.bamboo as boo
import pingparser.runningval as runningval

VERSION = "lorena_v01"
DATE = "10.18.23"
ORIGINAL_NAME = "event_extractors/lorena.py"


COLNAMES = ['TrialNum', 'FixationDur', 'RespError_cuefrac',
            'Cue_D', 'CueGoneDist_cuefrac', 'CueMove',
            'RespPause', 'RespBox_type',
            'CueRel1_x', 'CueRel1_y', 'CueRel2_x', 'CueRel2_y',
            'Cue1_x', 'Cue1_y', 'Cue2_x', 'Cue2_y',
            'stick_angle1', 'stick_angle2',
            'stick1_x', 'stick1_y', 'stick2_x', 'stick2_y',
            'stickhead1_x', 'stickhead1_y', 'stickhead2_x', 'stickhead2_y',
            'Welzl_x', 'Welzl_y', 'Welzl_D',
            'WMDelay', 'WMTrial']


#list of dictionary for slicing away raw df (with bamboo.slice, polarity '-), in pre_processor()
RAW_TO_DELETE = [{'Subject':['RespError_cuefrac'],'TrialNum':[1], 'Value':['0']}]

#list of running values to grab using running val
RUNNING_VALS = ['Cue_D', 'CueGoneDist_cuefrac','FixationDur','RespPause','WMDelay']


def pre_processor(df_sess_raw):
    df = df_sess_raw.copy()

    for r in RAW_TO_DELETE:
        df = boo.slice(df, r, '-')

    return df

def post_processor(df):
    #1. create variable for CueMove
    df['CueMove'] = (df['Cue1_x'] == df['Cue2_x']) & (df['Cue1_y'] == df['Cue2_y']) == False

    #2. convert WMTrial variable from string to boolean
    df['WMTrial'] = df['WMTrial']=='True'
    return df

def running_valuator(df_sess_raw):
    #extract running values (cannot be done separately using trial summarizer)
    vals_all = runningval.get(df_sess_raw, RUNNING_VALS, debug=False)
    return vals_all


def trial_summarizer(df_trial):
    """
    df_trial: df containing all events restricted to a single trial.
    return crucial trial info as a single row
    crucial trial info differs across experiments
    """

    fixated = boo.slice(df_trial, {'Value': ['FixationCompleted']}, '+')  # index where InTimerCompleted

    if len(fixated) != 1:  # TODO account for trials where fixation did not complete, there was more than 1 FixationCompleted (bug/error)
        return None

    fixated_row = fixated.index[0]


    row_holder = pd.DataFrame({'val': None}, index=COLNAMES)

    row_holder.loc['TrialNum', 'val'] = df_trial['TrialNum'].iloc[0]

    for param in ['RespError_cuefrac']:
        row_holder.loc[param, 'val'] = genparse.get_trial_param(df_trial, param, dtype='float', single='strict')

    for param in ['Welzl_D']:
        row_holder.loc[param, 'val'] = genparse.get_trial_param(df_trial, param, dtype='float', single='last')


    #extract xy subjects that are not prepost (below)
    for xy_param in ['Welzl']:
        xy_str = genparse.get_trial_param(df_trial, xy_param+'_xy',
                                          dtype='string', single='last')
        if xy_str is None: #not found
            xy = (None, None)
        else:
            xy = genparse.str_to_list(xy_str)

        row_holder.loc[xy_param + '_x'] = xy[0]
        row_holder.loc[xy_param + '_y'] = xy[1]



    for param in ['RespBox_type','WMTrial']:
        row_holder.loc[param, 'val'] = genparse.get_trial_param(df_trial, param, dtype='string', single='last')

    # === params with xy, and pre and post fixation ===
    for param, prefix in zip(['Cue_xy', 'stickhead_xy', 'stick_xy', 'Cue_xyRel'],
                             ['Cue'   , 'stickhead',    'stick',    'CueRel']):

        param_prepost = genparse.closest_to_row(df_trial, param, fixated_row)

        # TODO: can improve for efficiency
        for i, v in enumerate(param_prepost):
            if v is None:
                param_prepost[i] = '(nan,nan)'

        output = genparse.format_prepost_xy(param_prepost, prefix)

        for k, v in output.items():
            row_holder.loc[k, 'val'] = v

    # === params with pre and post fixation ===
    for param in ['stick_angle']:
        param_prepost = genparse.closest_to_row(df_trial, param, fixated_row)

        for i, v in enumerate(param_prepost):
            name = f'{param}{i + 1}'
            row_holder.loc[name, 'val'] = v



    return row_holder