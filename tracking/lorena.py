#TODO: harmonize this with touch.py

BONSAI_TIMESTAMP_FMT = "%H:%M:%S.%f"

import pingparser.general as genparse
import pandas as pd, numpy as np
import pyfun.bamboo as boo, pyfun.timestrings as timestr

VERSION = "lorena_v01"
DATE = "10.18.23"
ORIGINAL_NAME = "tracking/lorena.py"


def resp_xy(fn, sess_name, tracker_name):
    """
    process response tracking for a single session csv
    """

    xy_raw = genparse.read_raw(fn) #tracking xy file e.g. snout_xy

    # ==== extract basic response markers ===
    # 1. start of response period to consider ('RespMod_on'),
    # 2. end of response period to consider  ('Trial_ended'),
    # 3. first counted response in zone
    # these go into "row_key"--row numbers in xy_raw corresponding to the marker events

    df = boo.slice(xy_raw, {'Subject': ['ModEvent'],
                               'Value': ['RespMod_on']})

    df_start = boo.swap_col_with_index(df, 'TrialNum', 'start')

    df = boo.slice(xy_raw, {'Subject': ['Trial_ended'],
                               'Value': ['True']})

    df_end = boo.swap_col_with_index(df, 'TrialNum', 'end')

    # extract rows where the first touch was completed (touched within response zone)

    df = boo.slice(xy_raw, {'Subject': ['RespEvent'],
                               'Value': ['RespMade']})

    df_resp_made = boo.swap_col_with_index(df, 'TrialNum', 'RespBox_touched')

    # combine response markers into single table
    # TODO: consider including incomplete trials (currently excluded because it is an inner join)
    row_key = boo.merge_by_index([df_start, df_end, df_resp_made], join='inner')

    # sanity check
    is_sane = (row_key['end'] > row_key['RespBox_touched']) * (row_key['RespBox_touched'] > row_key['start'])
    if not is_sane.all():
        raise ValueError('Sanity check fail')

    if not (row_key['start'].diff()[1:] > 0).all():
        raise ValueError('Row numbers not monotonically increasing?')

    resp_xy_df = pd.DataFrame([], columns=['x', 'y', 'TrialNum', 'sess'])

    # ===extract touch response per trial ===
    for TrialNum, row in row_key.iterrows():
        trial_rows = xy_raw.loc[row['start']:row['end']]

        if len(trial_rows) == 0:
            raise ValueError('No rows found. This is not possible, because of the way row_key was created')

        trial_rows = boo.slice(trial_rows, {'Subject': [tracker_name]})[['Value', 'Timestamp']]

        trial_rows['Timestamp'] = timestr.parse_time_col(trial_rows['Timestamp'])  # convert strings to timestamps
        trial_rows = genparse.str_to_list_col(trial_rows, 'Value', 'x', 'y')  # convert xy_str to x and y columns

        # === find the row that corresponds to the counted touch (first touch within resp box) ===
        # we assume that the touch that was counted, was the one that occurred immediately before Touch_RespBox event
        mask = trial_rows.index < (row['RespBox_touched'])
        try:
            counted_row = trial_rows[mask].iloc[-1, :]
        except IndexError:
            print('No touch xy found that corresponds to counted touch!')
            raise

        trial_rows = timestr.calc_dt_col(trial_rows, 'Timestamp', 'dt_ms', counted_row['Timestamp'])
        # convert timestamps into time relative to first counted touch

        trial_rows['TrialNum'] = TrialNum
        resp_xy_df = resp_xy_df.append(trial_rows)

    resp_xy_df['sess'] = sess_name

    return resp_xy_df


def cam2bs(df: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    # TODO: clean up cam2bs_mat
    # Ensure that the input dataframe has 'x' and 'y' columns
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("Input dataframe must contain 'x' and 'y' columns.")

    # Transformation matrix
    cam2bs_mat = np.array([
        [-9.639814839905778, 0.04355386876169842, -0.24301057903562098],
        [0.013579130748207479, 11.850996784005718, -0.20830413790599578],
        [-0.0465170028745135, 0.2072642741558346, 0.9951820131724353]
    ])

    cam2bs_mat = np.array([
        [-9.639814839905778, -9.639814839905778, -0.24301057903562098],
        [0.013579130748207479, 11.850996784005718, -0.20830413790599578],
        [-0.0465170028745135, 0.2072642741558346, 0.9951820131724353]
    ])


    # Creating homogeneous coordinates without altering original df
    homogeneous_coords = np.c_[df[['x', 'y']].values, np.ones(len(df))]

    # Performing matrix multiplication
    result = homogeneous_coords @ cam2bs_mat.T

    # Depending on overwrite option, either overwrite original columns or create new ones
    if overwrite:
        df['x'] = result[:, 0] / result[:, 2]
        df['y'] = result[:, 1] / result[:, 2]
    else:
        df['x_bs'] = result[:, 0] / result[:, 2]
        df['y_bs'] = result[:, 1] / result[:, 2]

    return df