import numpy as np
import pandas as pd

class poke_permuter():
    """
    permutation testing for nose pokes to cue position on stick
    """
   
    def __init__(self, df):
        df = df.copy()
        
        #drop these to avoid confusion
        #these are CueRel defined at run time, specified to other coordinates
        df = df.drop(['CueRel1_x', 'CueRel1_y', 'CueRel2_x', 'CueRel2_y'], axis=1)
        
        
        #default column names
        self.rel_poke = {'x': 'Welzl_x_rel', 'y': 'Welzl_y_rel'}
        self.poke     = {'x': 'Welzl_x_bs',  'y': 'Welzl_y_bs'}
        self.target   = {'x': 'Cue2_x',      'y': 'Cue2_y'}
        self.rel_target = {'x': 'Cue_x_rel', 'y': 'Cue_y_rel'}
        self.ref      = {'x': 'stick2_x',    'y': 'stick2_y'}
        
        self.rel_poke_shuf = {'x':'Welzl_x_rel_shuf', 'y':'Welzl_y_rel_shuf'}
        self.poke_shuf     = {'x':'Welzl_x_shuf',     'y':'Welzl_y_shuf'}           
        
        self.target_size = 0.18
        
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

    def check_shuffles(self, df=None):
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

    def calc_err(self):
        err = self.calc_dists_y(self.rel_poke_xy, self.rel_tgt_xy) / self.target_size
        
        return err        
            
    def calc_shuffled_err_coarse(self):
        
        shuffle_indices = np.random.permutation(self.n)
        rel_poke_xy_shuf = self.rel_poke_xy[shuffle_indices]
        
        err = self.calc_dists_y(rel_poke_xy_shuf, self.rel_tgt_xy) / self.target_size 
        
        
        return err

    def calc_shuffled_err(self):
        # Get unique dates and the start index for each date's records
        unique_dates, indices = np.unique(self.df['date'], return_index=True)
        # Sort the indices to maintain the original date order
        sorted_indices = np.argsort(indices)
        sorted_dates = unique_dates[sorted_indices]

        # Shuffle within subsets defined by dates
        for date in sorted_dates:
            # Find the indices for rows corresponding to the current date
            date_indices = df[df['date'] == date].index
            # Convert DataFrame indices to positions in the matrix
            positions = np.where(np.isin(df.index, date_indices))[0]
            # Shuffle these positions
            np.random.shuffle(rel_poke_matrix[positions])

        return err

    def calc_dists(self, A, B):
        #calculate pair-wise euclidean distances
        #A & B are n x 2 matrices, which are n rows of x,y entries
        
        return np.linalg.norm(A - B, axis=1)
        

    def calc_dists_y(self, A, B):
        #calculate pair-wise euclidean distances
        #A & B are n x 2 matrices, which are n rows of x,y entries
        
        A[:,0]=0
        B[:,0]=0
        
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
                               'new_ids': date_indices.copy()}

        self.date_info = date_info


    def shuffle_within_date(self,group):
        # Shuffling only 'rel_poke_x' and 'rel_poke_y' within the group and returning the entire group
        shuffled_x = group[self.rel_poke['x']].sample(frac=1).reset_index(drop=True)
        shuffled_y = group[self.rel_poke['y']].sample(frac=1).reset_index(drop=True)

        # Assigning the shuffled values back to the group
        group[self.rel_poke['x']] = shuffled_x
        group[self.rel_poke['y']] = shuffled_y

        return group

class sim_permuter(poke_permuter):
    def __init__(self, ntrials=100, datechunk_M=20, datechunk_SD=10):
        # create simulated data

        sim = {'Welzl_x_bs': [0] * ntrials,
               'Cue2_x': np.random.uniform(-1, 1, ntrials),
               'stick2_x': [0] * ntrials}

        sim_df = pd.DataFrame(sim)

        for c in ['Welzl_y_bs', 'Cue2_y', 'stick2_y', 'CueRel1_x', 'CueRel1_y', 'CueRel2_x', 'CueRel2_y']:
            sim_df[c] = 0

        sim_df['Welzl_x_bs'] = sim_df['Cue2_x'] + np.random.normal(loc=0.0, scale=1,
                                                                   size=ntrials)

        sim_df['Welzl_y_bs'] = sim_df['Cue2_y'] + np.random.normal(loc=0.0, scale=1,
                                                                   size=ntrials)


        # sim_df['Welzl_x_bs']=np.random.normal(loc=0.0,scale=0.01,size=ntrials)
        # sim_df.loc[sim_df['Welzl_x_bs']>1,['Welzl_x_bs']]=1
        # sim_df.loc[sim_df['Welzl_x_bs']<-1,['Welzl_x_bs']]=-1

        sim_df = self.generate_dates(sim_df, M=datechunk_M, SD=datechunk_SD)

        super().__init__(sim_df)


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



