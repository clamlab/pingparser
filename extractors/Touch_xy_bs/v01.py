#TODO: harmonize with lorena tracking_top. the two are identical except for the marker indicating counted touch


#this is in bonsai coordinates


BONSAI_TIMESTAMP_FMT = "%H:%M:%S.%f"

import pingparser.general as genparse
import pandas as pd
import pyfun.bamboo as boo, pyfun.timestrings as timestr

VERSION = "touch_v01"
DATE = "11.03.23"
ORIGINAL_NAME = "tracking_top/touch_v01.py"


def resp_xy(fn, sess_name, tracker_name):
    """
    process touch for a single session csv
    """

    touch_raw = genparse.read_raw(fn)

    # ==== extract basic response markers ===
    # 1. start of response period to consider ('RespMod_on'),
    # 2. end of response period to consider  ('Trial_ended'),
    # 3. first counted response in zone ('Touch_RespBox=True')
    # these go into "row_key"--row numbers in touch_raw corresponding to the marker events

    df = boo.slice(touch_raw, {'Subject': ['ModEvent'],
                               'Value': ['RespMod_on']})

    df_start = boo.swap_col_with_index(df, 'TrialNum', 'start')

    df = boo.slice(touch_raw, {'Subject': ['Trial_ended'],
                               'Value': ['True']})

    df_end = boo.swap_col_with_index(df, 'TrialNum', 'end')

    # extract rows where the first touch was counted (touched within response zone)

    df = boo.slice(touch_raw, {'Subject': ['Touch_RespBox'],
                               'Value': ['True']})

    df_touched = boo.swap_col_with_index(df, 'TrialNum', 'RespBox_touched')

    # combine response markers into single table
    # TODO: consider including incomplete trials (currently excluded because it is an inner join)
    row_key = boo.merge_by_index([df_start, df_end, df_touched], join='inner')

    # sanity check
    is_sane = (row_key['end'] > row_key['RespBox_touched']) * (row_key['RespBox_touched'] > row_key['start'])
    if not is_sane.all():
        raise ValueError('Sanity check fail')

    if not (row_key['start'].diff()[1:] > 0).all():
        raise ValueError('Row numbers not monotonically increasing?')

    #touch_df = pd.DataFrame([], columns=['x', 'y', 'TrialNum', 'sess'])
    touch_df = pd.DataFrame({
        'x': pd.Series(dtype='float64'),  # x and y as float
        'y': pd.Series(dtype='float64'),
        'TrialNum': pd.Series(dtype='int64'),  # TrialNum as int
        'sess': pd.Series(dtype='object')  # sess as string (object type)
    })


    # ===extract touch response per trial ===
    for TrialNum, row in row_key.iterrows():
        touch_rows = touch_raw.loc[row['start']:row['end']]

        if len(touch_rows) == 0:
            raise ValueError('No rows found. This is not possible, because of the way row_key was created')

        touch_rows = boo.slice(touch_rows, {'Subject': [tracker_name]})[['Value', 'Timestamp']]

        touch_rows['Timestamp'] = timestr.parse_time_col(touch_rows['Timestamp'])  # convert strings to timestamps
        touch_rows = genparse.str_to_list_col(touch_rows, 'Value', 'x', 'y')  # convert xy_str to x and y columns

        # === find the row that corresponds to the counted touch (first touch within resp box) ===
        # we assume that the touch that was counted, was the one that occurred immediately before Touch_RespBox event
        mask = touch_rows.index < (row['RespBox_touched'])
        try:
            counted_row = touch_rows[mask].iloc[-1, :]
        except IndexError:
            print('No touch xy found that corresponds to counted touch!')
            raise

        touch_rows = timestr.calc_dt_col(touch_rows, 'Timestamp', 'dt_ms', counted_row['Timestamp'])
        # convert timestamps into time relative to first counted touch

        touch_rows['TrialNum'] = TrialNum
        if len(touch_df)==0:
            pass
        print(len(touch_df))
        touch_df = pd.concat([touch_df, touch_rows])


    touch_df['sess'] = sess_name

    return touch_df