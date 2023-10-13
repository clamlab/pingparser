"""
event parser for pSWM Lorena setup
"""

COLNAMES = ['TrialNum', 'FixationDur', 'RespError_cuefrac', 'Cue_D', 'CueGoneDist_cuefrac',
            'CueRel1_x', 'CueRel1_y', 'CueRel2_x', 'CueRel2_y',
            'Cue1_x', 'Cue1_y', 'Cue2_x', 'Cue2_y',
            'stick1_x', 'stick1_y', 'stick2_x', 'stick2_y', 'WMDelay',
            'WMTrial']

import pandas as pd
from ..import general as genparse
import pyfun.bamboo as boo



def trial_summary(df_trial):
    """
    df_trial: df containing all events restricted to a single trial.
    return crucial trial info as a single row
    crucial trial info differs across experiments
    """

    fixated = boo.slice(df_trial, {'Value': ['FixationCompleted']}, '+')  # index where InTimerCompleted

    if len(fixated) == 0:  # TODO account for trials where fixation did not complete
        return None

    fixated_row = fixated.index[0]


    row_holder = pd.DataFrame({'val': None}, index=COLNAMES)

    row_holder.loc['TrialNum', 'val'] = df_trial['TrialNum'].iloc[0]

    for param in ['RespError_cuefrac']:
        row_holder.loc[param, 'val'] = genparse.get_trial_param(df_trial, param, dtype='float')

    for param in ['WMTrial']:
        row_holder.loc[param, 'val'] = genparse.get_trial_param(df_trial, param, dtype='string')


    for param, prefix in zip(['Cue_xy', 'stick_xy', 'Cue_xyRel'],
                             ['Cue'   , 'stick',    'CueRel']):

        param_prepost = genparse.closest_to_row(df_trial, param, fixated_row)

        # TODO: can improve for efficiency
        for i, v in enumerate(param_prepost):
            if v is None:
                param_prepost[i] = '(nan,nan)'

        output = genparse.format_prepost_xy(param_prepost, prefix)

        for k, v in output.items():
            row_holder.loc[k, 'val'] = v

    return row_holder