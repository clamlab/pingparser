import numpy as np
import pandas as pd
from pyfun.stats import select_averager
import matplotlib.gridspec as gridspec
from panplots import plotters

def insert_xy(input_string):
    """
    For input string e.g. abc_$_def, return dict
    {'x': 'abc_x_def',
     'y': 'abc_y_def'}
    """
    xy_string = {}
    for coord in ['x', 'y']:
        xy_string[coord] = input_string.replace('$', coord)
    return xy_string



class poke_permuter():
    """
    permutation testing for nose pokes to cue position on stick
    """
   
    def __init__(self, df, averager_type, xy_col_names, date_col='date'):
        """
        :param xy_col_names: dict with 'poke', 'ref', 'target' as keys,
                          and single string value with a "$" placeholder

        :param averager_type: averaging function to apply to errors. Either 'median' or 'mean'
        """

        df = df.copy()
        self.date_col = date_col #column name for shuffling by date


        #drop these to avoid confusion
        #these are CueRel defined at run time, specified to other coordinates
        '''
        try:
            df = df.drop(['CueRel1_x', 'CueRel1_y', 'CueRel2_x', 'CueRel2_y'], axis=1)
        except KeyError:
            pass
        '''

        self.averager = select_averager(averager_type)

        self.df = df
        
        # === define column names to look for ===
        self.poke   = insert_xy(xy_col_names['poke'])
        self.ref    = insert_xy(xy_col_names['ref'])
        self.target = insert_xy(xy_col_names['target'])

        # === define column names to create ===
        self.rel_poke = insert_xy('poke_$_rel')
        self.rel_target = insert_xy('target_$_rel')
        self.rel_poke_shuf = insert_xy('poke_$_rel_shuf')

        # check that new column names don't clash with existing columns
        for c_group in [self.rel_poke, self.rel_target, self.rel_poke_shuf]:
            for c in c_group.values():
                if c in self.df.columns:
                    raise ValueError('Error! Column name ' + c + ' already exists!')


        self.update()



    def update(self):
        df = self.df
        # === compute coordinates relative to reference ===
        # these are stored in df to debug / compare within original df,
        # but the shuffle computations are performed on np matrices (below)
        self.calc_rel2ref(df, old_labels = self.poke,   new_labels = self.rel_poke)
        self.calc_rel2ref(df, old_labels = self.target, new_labels = self.rel_target)

        #sanity checks
        self.check_nan(df)
        self.check_dates(df)

        self.df = df
        self.grab_date_info()

        #get numpy matrices (faster compute on matrices instead of dfs)
        self.rel_poke_xy = df[[  self.rel_poke['x'],   self.rel_poke['y']]].to_numpy()
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
        if df[self.date_col].is_monotonic_increasing is not True:
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

        for date, group in df.groupby(self.date_col):
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

    def calc_dists(self, A, B, method='euclid'):
        #calculate pair-wise distances, either euclid, or only in x or y direction
        #A & B are n x 2 matrices, which are n rows of x,y entries

        if method=='euclid':
            return np.linalg.norm(A - B, axis=1)
        elif method=='x':
            return np.abs(A[:,0]-B[:,0])
        elif method=='y':
            return np.abs(A[:,1]-B[:,1])
        else:
            raise ValueError('Invalid distance method ' + str(method))

    def grab_date_info(self):
        # for each session date, grab the row indices, and n
        # this will be computed once, and quickly applied (only) for within-session shuffles
        # where within-sess shuffle will just scramble the order of old_ids to form new_ids

        unique_dates = np.unique(self.df[self.date_col])
        date_info = {}
        for date in unique_dates:
            # Find row indices (note: iloc, not loc), to use for shuffling in shufonce_bysess()
            date_indices = np.where(self.df[self.date_col] == date)[0]

            date_info[date] = {'old_ids': date_indices.copy(),
                               'new_ids': date_indices.copy(),
                               'n':       len(date_indices)   }

        self.date_info = date_info

    def grab_err(self, dist_method):

        err = self.calc_dists(self.rel_poke_xy, self.rel_tgt_xy,  dist_method)

        return err

    def grab_err_by_date(self, dist_method):

        err_full = self.grab_err(dist_method)

        err = {}  # container for actual errors by date

        for date, v in self.date_info.items():
            err[date] = err_full[v['old_ids']]

        return err

    def permtest_coarse(self, n_shuffles, dist_method):
        err = self.grab_err(dist_method)
        err_M = self.averager(err)

        shuf_err_M = [] #container for shuffled errors
        for i in range(n_shuffles):
            shuf_err = self.shufonce_coarse(dist_method)
            shuf_err_M.append(self.averager(shuf_err))

        p_value = np.mean(shuf_err_M <= err_M)

        return p_value, shuf_err_M

    def permtest_fine(self, n_shuffles, dist_method):
        err = self.grab_err(dist_method)
        err_M = self.averager(err)

        shuf_err_M = [] #container for shuffled errors
        for i in range(n_shuffles):
            shuf_err = self.shufonce_fine(dist_method)
            shuf_err_M.append(self.averager(shuf_err))

        p_value = np.mean(shuf_err_M <= err_M)

        return p_value, shuf_err_M


    def permtest_bysess(self, n_shuffles, dist_method):

        err = self.grab_err_by_date(dist_method)
        err_M = {k: self.averager(v) for k, v in err.items()}

        shuf_err_M = {k: [] for k in err.keys()}  #container for shuffled errors

        for i in range(n_shuffles):
            shuf_err = self.shufonce_bysess(dist_method)

            for date, this_shuf_err in shuf_err.items():
                this_shuf_err_M = self.averager(this_shuf_err)
                shuf_err_M[date].append(this_shuf_err_M)

        # calculate p-values
        p_value = {}
        for date, this_err in err_M.items():
            p_value[date] = np.mean(shuf_err_M[date] <= this_err)

        return p_value, shuf_err_M


    def shufonce_coarse(self ,dist_method):
        #shuffle entire dataset, across sessions, and calculate error
        shuffle_indices = np.random.permutation(self.n)
        self.rel_poke_xy_shuf = self.rel_poke_xy[shuffle_indices]

        shuf_err = self.calc_dists(self.rel_poke_xy_shuf, self.rel_tgt_xy, dist_method)

        return shuf_err

    def shufonce_fine(self, dist_method):
        #shuffle entire dataset, but per-sessions, and calculate error
        self.shuffle_by_sess()
        shuf_err = self.calc_dists(self.rel_poke_xy_shuf, self.rel_tgt_xy, dist_method)

        return shuf_err

    def shufonce_bysess(self, dist_method):
        # shuffle entire dataset, but per-sessions, and calculate error per session
        self.shuffle_by_sess()

        shuf_err = {}

        for date, v in self.date_info.items():

            A =       self.rel_tgt_xy[v['old_ids'], :]
            B = self.rel_poke_xy_shuf[v['old_ids'], :]

            shuf_err[date] = self.calc_dists(A, B, dist_method)

        return shuf_err

    def shuffle_by_sess(self):
        #shuffle by sess without calculating error

        df = self.df

        for v in self.date_info.values():
            np.random.shuffle(v['new_ids'])
            self.rel_poke_xy_shuf[v['old_ids'], :] = self.rel_poke_xy[v['new_ids'], :]

class sim_permuter(poke_permuter):
    def __init__(self, averager_type, ntrials=100, datechunk_M=20, datechunk_SD=10):


        xy_col_names = {'poke':   'Welzl_$_bs',
                        'ref' :   'stick2_$',
                        'target': 'Cue2_$' }
        date_col = 'date'

        # === create container for simulated data ===
        sim = {'Welzl_x_bs': np.zeros(ntrials), #poke
               'Welzl_y_bs': np.zeros(ntrials),
               'Cue2_x'    : np.zeros(ntrials), #target
               'Cue2_y'    : np.zeros(ntrials),
               'stick2_x'  : np.zeros(ntrials), #ref
               'stick2_y'  : np.zeros(ntrials)}

        sim_df = pd.DataFrame(sim)




        sim_df = self.generate_dates(sim_df, M=datechunk_M, SD=datechunk_SD)


        super().__init__(sim_df, averager_type, xy_col_names, date_col)

        self.update_sim()

    def update_sim(self):
        A = np.array(self.df[['Welzl_x_bs', 'Welzl_y_bs']])
        B = np.array(self.df[['Cue2_x', 'Cue2_y']])
        self.df['RespError_cuefrac'] = np.linalg.norm(A - B, axis=1) #not used in shuffling, just for sanity check

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


def plot_shuffle_battery(fig, plot_title, permer, n_shuffles, date_labels=False):
    """
    example usage:
    plot a battery of shuffle tests
    """


    # === set up subplots layout ===
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.25)

    fig.text(0.5, 0.95, plot_title, ha='center', va='top')
    axes = list(range(4))

    axes[0] = (fig.add_subplot(gs[0, :]))  # First row, spanning all columns

    for i in range(1, 4):
        axes[i] = fig.add_subplot(gs[1, i - 1])  # Second row, first column

    for ax in axes:
        ax.spines[['right', 'top']].set_visible(False)

    ax = axes[0]
    sess_pvals_all = {}
    for m in ['euclid', 'x', 'y']:
        sess_pvals, null_dist = permer.permtest_bysess(n_shuffles, dist_method=m)
        ax.plot(sess_pvals.keys(),sess_pvals.values(), label=m)

        sess_pvals_all[m] = sess_pvals

    if date_labels:
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.set_xticklabels([])

    ax.set_xlabel('Session')
    ax.set_ylabel('p-value')

    ax.axhline(0.05, color='k', linewidth=1, linestyle='--')
    legend = ax.legend(loc=(.9,.9))
    legend.get_frame().set_alpha(0.2)

    p = {}
    for m in ['euclid', 'x', 'y']:
        p[m], null_dist = permer.permtest_fine(n_shuffles, dist_method=m)
        p[m] = str(p[m].round(2))

    txtbox = 'Fine' + '\nxy: ' + p['euclid'] + '\n x: ' + p['x'] + '\n y: ' + p['y']

    ax.text(0.05, 0.9, txtbox, fontsize=10, transform=ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 5})

    p = {}
    for m in ['euclid', 'x', 'y']:
        p[m], null_dist = permer.permtest_coarse(n_shuffles, dist_method=m)
        p[m] = str(p[m].round(2))

    txtbox = 'Coarse' + '\nxy: ' + p['euclid'] + '\n x: ' + p['x'] + '\n y: ' + p['y']
    ax.text(0.1, 0.9, txtbox, fontsize=10, transform=ax.transAxes,
            bbox={'facecolor': 'white', 'alpha': 0.3, 'pad': 5})

    # === histograms for session p values ===

    for i, m in enumerate(['euclid', 'x', 'y']):
        ax = axes[i + 1]

        pvals = np.array(list(sess_pvals_all[m].values()))

        ax.hist(pvals, bins=100);

        pvals = np.array(list(sess_pvals_all[m].values()))
        sig_frac = np.mean(pvals < 0.05) * 100  # fraction of data that are p < .05
        txt = str(sig_frac.round(3)) + '% with p < .05'

        ax.text(0.7, 0.1, txt, fontsize=12, transform=ax.transAxes,
                bbox={'facecolor': 'yellow', 'alpha': 0.5, 'pad': 10})

        ax.set_title(m)
        if i == 1:
            ax.set_xlabel('p_value')
        ax.axvline(0.05, color='k', linewidth=1, linestyle='--')
