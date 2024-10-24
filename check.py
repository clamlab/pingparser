"""
sanity checkers
"""

import pyfun.timestrings as timestr
import random
import pyfun.bamboo as boo
from . import general as genparse
from . import runningval
import numpy as np



def random_trial(anim, running_subj, mode='big'):
    """
    given some anim, randomly pick a session, and a trial
                  for sanity checking

    running_subj: names of subjects to print the most recent value (on current trial, or closest previous trial)
                    note--this doesn't care whether value occurred before or after fixation

    mode: big returns random row from big_df
          new returns random row from new_df
          subsess random selects a subsess_df

    output: prints row summary of randomly-selected trial
            saves raw eventpings of selected trial, and previous, into foo.csv
            prints running values: [val on current trial if found, trial number of found recent val, val]
                --> NB. if running value found for current trial, could have occurred b4/after fixation, so check raw

    """

    print("*** LOOK IN foo.csv for raw trial pings ***")
    print()

    if mode=='subsess':
        df = random.choice(list(anim.subsess.values()))
    elif mode=='big':
        df = anim.big_df
    elif mode=='new':
        df = anim.new_df
    else:
        error_str = 'Mode '+ mode+ ' not found. Only have big or subsess.'
        raise ValueError(error_str)

    if len(df)==0:
        print('Dataframe empty')
        return

    row = df.sample(n=1).iloc[0]
    subsess_name = row['sess']

    print(anim.name, subsess_name)
    print()

    fn = anim.subsess_paths[subsess_name]['Events']
    df_sess_raw = genparse.read_raw(fn)

    trial_num = row['TrialNum']

    df_trial_raw = boo.slice(df_sess_raw, {'TrialNum': [trial_num - 1, trial_num]})
    df_trial_raw.to_csv('foo.csv', index=False)

    vals_all = {}
    for subj_name in running_subj:
        vals_all[subj_name] = runningval.find_recent(df_sess_raw, subj_name, trial_num)

    print(row)
    print()
    print(vals_all)

    return row


def new_df_lens(expt):
    all_passed = True
    err_msgs = []
    for anim_name, anim in expt.anim.items():
        new_lens = len(anim.new_df)
        sess_lens = np.sum([len(df) for df in anim.sess.values()])
        subsess_lens = np.sum([len(df) for df in anim.subsess.values()])
        passed = new_lens == sess_lens == subsess_lens

        if passed == False:
            all_passed = False
            err_msg = f"{anim_name}, big: {new_lens}, sess: {sess_lens}, subsess: {subsess_lens}"
            print(err_msg)
            err_msgs.append(err_msg)

    if all_passed:
        print('Overall length checks passed.')

    return err_msgs

def subsess_lens(expt):
    # ===check that lengths of subsess_dfs are preserved
    # when merged with other same-day subsess ===

    #input: expt

    any_errors = False

    err_msgs = []

    for anim_name, anim in expt.anim.items():

        for sess_name, sess_df in anim.sess.items():
            if len(sess_df) > 0:
                subsess_n_all = sess_df['sess'].value_counts()
            else:
                subsess_n_all = None

            subsess_on_date = timestr.filter_by_date(anim.subsess, sess_name)

            for subsess_df in subsess_on_date.values():

                n_actual = len(subsess_df)
                if n_actual == 0:
                    continue

                subsess_name = subsess_df.sess.iloc[0]

                if subsess_name in subsess_n_all:
                    n_in_merged = subsess_n_all[subsess_name]
                else:
                    n_in_merged = 0

                if n_in_merged!=n_actual:
                    err_msg = f"{anim_name}, {subsess_name} is {n_actual} vs {n_in_merged}"
                    print(err_msg)
                    err_msgs.append(err_msg)
                    any_errors = True

    if any_errors==False:
        print('Subsess length checks passed.')

    return err_msgs
