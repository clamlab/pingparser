import numpy as np
import pandas as pd

class poke_permuter():
    """
    permutation testing for nose pokes to cue position on stick
    """
   
    def __init__(self, df, target_size):
        df = df.copy()
        
        #drop these to avoid confusion
        #these are CueRel defined at run time, specified to other coordinates
        df = df.drop(['CueRel1_x', 'CueRel1_y', 'CueRel2_x', 'CueRel2_y'], axis=1)
        self.df = df
        
        #default column names
        self.rel_poke = {'x': 'Welzl_x_rel', 'y': 'Welzl_y_rel'}
        self.poke     = {'x': 'Welzl_x_bs',  'y': 'Welzl_y_bs'}
        self.target   = {'x': 'Cue2_x',      'y': 'Cue2_y'}
        self.rel_target = {'x': 'Cue_x_rel', 'y': 'Cue_y_rel'}
        self.ref      = {'x': 'stick2_x',    'y': 'stick2_y'}
        
        self.rel_poke_shuf = {'x':'Welzl_x_rel_shuf', 'y':'Welzl_y_rel_shuf'}
        self.poke_shuf     = {'x':'Welzl_x_shuf',     'y':'Welzl_y_shuf'}           
        
        self.target_size = target_size

        self.update()

    def update(self):
        df = self.df
        #compute coordinates relative to reference
        self.calc_rel2ref(df, self.poke,   self.rel_poke)
        self.calc_rel2ref(df, self.target, self.rel_target)


        #sanity checks
        self.check_nan(df)
        self.check_dates(df)

        self.df = df

        self.grab_date_info()

        #get numpy matrices (faster compute?)
        self.rel_poke_xy = df[[self.rel_poke['x'], self.rel_poke['y']]].to_numpy()
        self.rel_tgt_xy =  df[[self.rel_target['x'], self.rel_target['y']]].to_numpy()
        self.rel_poke_xy_shuf = self.rel_poke_xy.copy() #to shuffle later
        self.n = len(self.rel_poke_xy)
        
        

    def calc_rel2ref(self, df, old_labels, new_labels):
        #===calculate coordinates relative to ref===
        for v in ['x', 'y']:
            df[new_labels[v]] = df[old_labels[v]] - df[self.ref[v]]        
            
    def check_nan(self, df):
        #===check for any NaNs===
        nan_instances = 0
        for col_pair in [self.rel_poke, self.poke, self.target, self.ref]:
            for c in col_pair.values():
                nan_instances += np.sum(df[c].isna())
        if nan_instances > 0:
            raise ValueError('ERROR! Some NaNs detected. Clean up your df')

    def check_dates(self, df):
        if df['date'].is_monotonic_increasing is not True:
            raise ValueError('ERROR! Dates should be monotonically increasing.')

    def check_insess_shuffles(self, df=None):
        # check that within session shuffling works

        if df is None:
            df = self.df.copy()

        #grab col names
        old_x, old_y =      self.rel_poke['x'],      self.rel_poke['y']
        new_x, new_y = self.rel_poke_shuf['x'], self.rel_poke_shuf['y']

        df[new_x] = self.rel_poke_xy_shuf[:, 0]
        df[new_y] = self.rel_poke_xy_shuf[:, 1]

        self.df = df

        bools_all = {}

        for date, group in df.groupby('date'):
            bools = {}
            # check that mean value across session, is preserved after shuffle
            # round the means to 10th decimal place first--i've found some small rounding differences
            # possibly from converting back and forth btw numpy and df

            bools['mean_x'] = group[[old_x, new_x]].mean().round(10).nunique() == 1  # 1 unique mean value
            bools['mean_y'] = group[[old_y, new_y]].mean().round(10).nunique() == 1

            bools['order_x'] = np.sum(np.abs((group[old_x] - group[new_x]))) != 0
            bools['order_y'] = np.sum(np.abs((group[old_y] - group[new_y]))) != 0

            bools_all[date] = bools

        #check pct sessions which pass all shuffle checks (mean and order, x and y)
        sess_bool = []
        for bools in bools_all.values():
            sess_bool.append(all([v for v in bools.values()]))

        pct_shuffled_sessions = np.mean(sess_bool) * 100

        print('% successfully shuffled: ', pct_shuffled_sessions)

        return bools_all

    def calc_dists(self, A, B):
        #calculate pair-wise euclidean distances
        #A & B are n x 2 matrices, which are n rows of x,y entries
        
        return np.linalg.norm(A - B, axis=1)

    def grab_date_info(self):
        # for each session date, grab the row indices, and n
        # this will be computed once, and quickly applied for each within-session shuffle

        unique_dates = np.unique(self.df['date'])
        date_info = {}
        for date in unique_dates:
            # Find row indices (note: iloc, not loc)
            date_indices = np.where(self.df['date'] == date)[0]

            date_info[date] = {'old_ids': date_indices.copy(),
                               'new_ids': date_indices.copy(),
                               'n':       len(date_indices)   }

        self.date_info = date_info

    def grab_err(self):
        err = np.array(self.df['RespError_cuefrac'])
        return err

    def grab_err_by_date(self):
        err_full = np.array(self.df['RespError_cuefrac'])
        err = {}  # container for actual errors by date

        for date, v in self.date_info.items():
            err[date] = err_full[v['old_ids']]

        return err

    def permtest_coarse(self, n_shuffles):
        err_M = np.mean(self.grab_err())
        shuf_err_M = [] #container for shuffled errors
        for i in range(n_shuffles):
            shuf_err_M.append(np.mean(self.shufonce_coarse()))

        p_value = np.mean(shuf_err_M <= err_M)

        return p_value, shuf_err_M

    def permtest_fine(self, n_shuffles):
        err_M = np.mean(self.grab_err())
        shuf_err_M = [] #container for shuffled errors
        for i in range(n_shuffles):
            shuf_err_M.append(np.mean(self.shufonce_fine()))

        p_value = np.mean(shuf_err_M <= err_M)

        return p_value, shuf_err_M


    def permtest_bysess(self, n_shuffles):

        err = self.grab_err_by_date()
        err_M = {k: np.mean(v) for k, v in err.items()}

        shuf_err_M = {k: [] for k in err.keys()}  #container for shuffled errors

        for i in range(n_shuffles):
            shuf_err = self.shufonce_bysess()

            for date, this_shuf_err in shuf_err.items():
                shuf_err_M[date].append(np.mean(this_shuf_err))

        # calculate p-values
        p_value = {}
        for date, this_err in err_M.items():
            p_value[date] = np.mean(shuf_err_M[date] <= this_err)

        return p_value, shuf_err_M


    def shufonce_coarse(self):
        #shuffle entire dataset, across sessions, and calculate error
        shuffle_indices = np.random.permutation(self.n)
        self.rel_poke_xy_shuf = self.rel_poke_xy[shuffle_indices]

        shuf_err = self.calc_dists(self.rel_poke_xy_shuf, self.rel_tgt_xy) / self.target_size

        return shuf_err

    def shufonce_fine(self):
        #shuffle entire dataset, but per-sessions, and calculate error
        self.shuffle_by_sess()
        shuf_err = self.calc_dists(self.rel_poke_xy_shuf, self.rel_tgt_xy) / self.target_size

        return shuf_err

    def shufonce_bysess(self):
        self.shuffle_by_sess()

        shuf_err = {}

        for date, v in self.date_info.items():

            A =       self.rel_tgt_xy[v['old_ids'], :]
            B = self.rel_poke_xy_shuf[v['old_ids'], :]

            shuf_err[date] = self.calc_dists(A, B) /  self.target_size

        return shuf_err

    def shuffle_by_sess(self, df=None):
        if df is None:
            df=self.df
        #shuffle rel pokes, but perform separate shuffle per date
        for v in self.date_info.values():
            np.random.shuffle(v['new_ids'])
            self.rel_poke_xy_shuf[v['old_ids'], :] = self.rel_poke_xy[v['new_ids'], :]

class sim_permuter(poke_permuter):
    def __init__(self, ntrials=100, datechunk_M=20, datechunk_SD=10):
        # create simulated data

        self.target_size = 0.18
        sim = {'Welzl_x_bs': np.random.uniform(low=-1,high=1,size=ntrials),
               'Welzl_y_bs': np.random.uniform(low=-1,high=1,size=ntrials),
               'Cue2_x': np.random.uniform(-1, 1, ntrials),
               'stick2_x': [0] * ntrials}

        sim_df = pd.DataFrame(sim)

        for c in ['Welzl_y_bs', 'Cue2_y', 'stick2_y', 'CueRel1_x', 'CueRel1_y', 'CueRel2_x', 'CueRel2_y']:
            sim_df[c] = 0

        #sim_df['Welzl_x_bs'] = sim_df['Cue2_x'] + np.random.normal(loc=0.0, scale=1,
        #                                                           size=ntrials)

        #sim_df['Welzl_y_bs'] = sim_df['Cue2_y'] + np.random.normal(loc=0.0, scale=1,
        #                                                           size=ntrials)


        # sim_df['Welzl_x_bs']=np.random.normal(loc=0.0,scale=0.01,size=ntrials)
        # sim_df.loc[sim_df['Welzl_x_bs']>1,['Welzl_x_bs']]=1
        # sim_df.loc[sim_df['Welzl_x_bs']<-1,['Welzl_x_bs']]=-1

        sim_df = self.generate_dates(sim_df, M=datechunk_M, SD=datechunk_SD)
        super().__init__(sim_df, self.target_size)

        self.update_sim()

    def update_sim(self):
        A = np.array(self.df[['Welzl_x_bs', 'Welzl_y_bs']])
        B = np.array(self.df[['Cue2_x', 'Cue2_y']])
        self.df['RespError_cuefrac'] = np.linalg.norm(A - B, axis=1) / self.target_size

        super().update()


    def generate_dates(self, df, M, SD, start_date='2024-01-01'):
        n = len(df)
        chunk_sizes = []
        while sum(chunk_sizes) < n:
            # Generate chunk size from a normal distribution and round to nearest int
            size = int(round(np.random.normal(M, SD)))
            if size > 0:
                chunk_sizes.append(size)

        # Adjust the last chunk size to fit the DataFrame exactly
        chunk_sizes[-1] = chunk_sizes[-1] - (sum(chunk_sizes) - n)

        # Generate chunk IDs
        chunk_ids = np.repeat(range(len(chunk_sizes)), chunk_sizes)
        df['chunk_id'] = chunk_ids[:n]  # In case the last adjustment made the array longer

        # Generate and assign dates
        dates = pd.date_range(start=start_date, periods=len(chunk_sizes), freq='D')
        chunk_id_to_date = {chunk_id: date for chunk_id, date in zip(range(len(chunk_sizes)), dates)}
        df['date'] = df['chunk_id'].map(chunk_id_to_date)


        # Drop the temporary chunk ID column
        df.drop('chunk_id', axis=1, inplace=True)
        df['date'] = df['date'].astype('str')

        return df



