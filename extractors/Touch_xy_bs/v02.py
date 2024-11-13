"""
Touch_xy_bs extractor for pSWM touchscreen setup.
"""

import pandas as pd
import pingparser.general as genparse
import pyfun.bamboo as boo
import pyfun.timestrings as timestr
import os


class Extractor:

    VERSION = "touch_v02"
    DATE = "11.08.23"
    TYPE = 'Touch_xy_bs'
    BONSAI_TIMESTAMP_FMT = "%H:%M:%S.%f"
    DF_TEMPLATE = pd.DataFrame([], columns=['TrialNum','Timestamp','x','y','sess'])


    def __init__(self):
        # Initialize any additional instance variables here if needed
        pass

    def extract(self, fn, sess_name):
        # Determine extraction method based on session date
        if timestr.search(sess_name)[1] < pd.to_datetime('2024-10-29'):
            extractor = self._extract_v_old
        else:
            extractor = self._extract_v_new

        return extractor(fn, sess_name)

    def _extract_v_old(self, fn, sess_name):
        # ==== Extract basic response markers ===
        # 1. Start of response period ('RespMod_on')
        # 2. End of response period ('Trial_ended')
        # 3. First counted response in zone ('Touch_RespBox=True')
        # These go into "row_key" - row numbers in df_sess_raw corresponding to the marker events

        df_sess_raw = genparse.read_raw(fn)  # Load raw data

        touch_df = boo.slice(df_sess_raw, {'Subject': [self.TYPE]})[['TrialNum', 'Value', 'Timestamp']]
        if len(touch_df)==0: #no touch entries
            return self.DF_TEMPLATE
        else:
            touch_df = genparse.str_to_list_col(touch_df, 'Value', 'x', 'y')  # Convert xy_str to x and y columns

        touch_df = touch_df.dropna(subset=None) #drop any rows where any of the col values are None / NaN

        touch_df['sess'] = sess_name

        return touch_df

    def _extract_v_new(self, fn, sess_name):
        df_sess_raw = genparse.read_raw(fn, colnames=["TrialNum", "x", "y", "Timestamp"])

        touch_df = df_sess_raw
        touch_df['sess'] = sess_name
        return touch_df





# Example usage for testing when running the module directly
if __name__ == "__main__":

    files = {'2024-10-20T10_40_14': "Y:/Edmund/Data/Touchscreen_pSWM/raw/EC05/2024_10_20-10_40/results_2024-10-20T10_40_14/Touch_xy_bs.csv",
             '2024-11-08T10_25_52': "Y:/Edmund/Data/Touchscreen_pSWM/raw/EC06/2024_11_08-10_25/results_2024-11-08T10_25_52/Touch_xy_bs.csv"}

    for subsess_name, fn in files.items():

        # Run extraction in parallel using ProcessPoolExecutor
        df_sess_raw = genparse.read_raw(fn)  # Load raw data

        # Check if the file exists
        if not os.path.exists(fn):
            print(f"Test file not found: {fn}")
            sys.exit(1)

        extractor = Extractor()

        # Run the extraction process on the test data
        touch_df = extractor.extract(fn, subsess_name)
        print(len(touch_df))


