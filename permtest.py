import numpy as np
import pandas as pd
from pyfun.stats import select_averager
import pyfun.stats

import matplotlib.gridspec as gridspec
import panplots.plotters as panplot

import copy

import matplotlib.lines as mlines
import matplotlib.pyplot as plt


from pyfun.customdict import invert_nested_dict

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

    def __init__(self, df, xy_col_names, metrics, min_trials_per_sess=2, date_col='date'):
        """
        :param xy_col_names: dict with 'poke', 'ref', 'target' as keys,
                          and single string value with a "$" placeholder

        :param metrics: dict with metric_obj instances
        """

        df = df.copy()
        self.date_col = date_col  # column name for shuffling by date


        self.df = df
        self.metrics = metrics
        self.min_trials_per_sess = min_trials_per_sess

        # === define column names to look for ===
        self.poke   = insert_xy(xy_col_names['poke'])
        self.ref    = insert_xy(xy_col_names['ref'])
        self.target = insert_xy(xy_col_names['target'])

        # === define column names to create ===
        self.rel_poke      = insert_xy('poke_$_rel')
        self.rel_target    = insert_xy('target_$_rel')
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
        self.calc_rel2ref(df, old_labels=self.poke, new_labels=self.rel_poke)
        self.calc_rel2ref(df, old_labels=self.target, new_labels=self.rel_target)

        # sanity checks
        self.check_nan(df)
        self.check_dates(df)

        self.df = df
        self.grab_date_info()

        # get numpy matrices (faster compute on matrices instead of dfs)
        self.rel_poke_xy = df[[  self.rel_poke['x'],   self.rel_poke['y']]].to_numpy()
        self.rel_tgt_xy  = df[[self.rel_target['x'], self.rel_target['y']]].to_numpy()
        self.rel_poke_xy_shuf = self.rel_poke_xy.copy()  # to shuffle later
        self.n = len(self.rel_poke_xy)

    def calc_dist_allmetrics(self, A, B):
        dist = {}
        for m_name, m in self.metrics.items():
            dist[m_name] = m.calc_dist(A, B)

        return dist

    def calc_avg_allmetrics(self, dist):
        """
        :param dist: dictionary of metric_name by dist arrays
        :return:
        """
        dist_avg = {}
        for m_name, m in self.metrics.items():
            dist_avg[m_name] = m.averager(dist[m_name])

        return dist_avg

    def calc_p_allmetrics(self, null, dist_avg):
        p_values = {}
        for m_name, m in self.metrics.items():
            p_values[m_name] = np.mean(m.null_hyp(null[m_name], dist_avg[m_name]))

        return p_values

    def calc_rel2ref(self, df, old_labels, new_labels):
        # ===calculate coordinates relative to ref===
        for v in ['x', 'y']:
            df[new_labels[v]] = df[old_labels[v]] - df[self.ref[v]]

    def check_nan(self, df):
        # ===check for any NaNs===
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

        # grab col names
        old_x, old_y = self.rel_poke['x'], self.rel_poke['y']
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

        # check pct sessions which pass all shuffle checks (mean and order, x and y)
        sess_bool = []
        for bools in bools_all.values():
            sess_bool.append(all([v for v in bools.values()]))

        pct_shuffled_sessions = np.mean(sess_bool) * 100

        print('% successfully shuffled: ', pct_shuffled_sessions)

        return bools_all


    def grab_date_info(self):
        # for each session date, grab the row indices, and n
        # this will be computed once, and quickly applied (only) for within-session shuffles
        # where within-sess shuffle will just scramble the order of old_ids to form new_ids

        unique_dates = np.unique(self.df[self.date_col])
        date_info = {}
        date_info_excluded = {}
        for date in unique_dates:
            # Find row indices (note: iloc, not loc), to use for shuffling in shufonce_bysess()
            date_indices = np.where(self.df[self.date_col] == date)[0]

            n_trials = len(date_indices)

            # exclude from shuffle analysis, dates which do not have minimum number of trials
            if n_trials < self.min_trials_per_sess:
                date_info_excluded[date] = {'n': n_trials}
                continue

            date_info[date] = {'old_ids': date_indices.copy(),
                               'new_ids': date_indices.copy(),
                               'n':       n_trials            }

        self.date_info          = date_info
        self.date_info_excluded = date_info_excluded

    def grab_dist(self, mode):
        """
        grab distances on actual data
        """
        if   mode=='data_all':
            dist = self.calc_dist_allmetrics(self.rel_poke_xy,      self.rel_tgt_xy)

        elif mode=='shuf_all':
            dist = self.calc_dist_allmetrics(self.rel_poke_xy_shuf, self.rel_tgt_xy)

        elif mode=='data_dated':
            dist = {}

            for date, v in self.date_info.items():
                dist[date] = self.calc_dist_allmetrics(self.rel_poke_xy[v['old_ids'], :],
                                                        self.rel_tgt_xy[v['old_ids'], :])
        elif mode=='shuf_dated':
            dist = {}
            for date, v in self.date_info.items():
                dist[date] = self.calc_dist_allmetrics(self.rel_poke_xy_shuf[v['old_ids'], :],
                                                             self.rel_tgt_xy[v['old_ids'], :])
        else:
            raise ValueError('Mode ' + mode + ' not found')

        return dist

    def permtest_all(self, n_shuffles, shuf_mode):
        """
        calculates one significance and distance value per dataset as oppose to per session
        However, can still shuffle within session or across whole dataset
        :param shuf_mode: either 'all' or 'by_date'

        """

        # === extract distances on data ===
        dist     = self.grab_dist('data_all')
        dist_avg = self.calc_avg_allmetrics(dist)  # apply averaging

        # === extract distances on shuffled data ===

        null = {key: [] for key in self.metrics.keys()} #container for null distributions by metric

        for i in range(n_shuffles):
            self.shuffle(shuf_mode)
            shuf_dist = self.grab_dist('shuf_all')
            shuf_dist_avg = self.calc_avg_allmetrics(shuf_dist)

            for m_name in self.metrics:
                null[m_name].append(shuf_dist_avg[m_name])

        # === calculate p values ===
        pvals = self.calc_p_allmetrics(null, dist_avg)

        return pvals, null



    def permtest_by_date(self, n_shuffles):
        # === extract distances on data ===
        dist_dated = self.grab_dist('data_dated') #dict of {date: {metric: dist} }

        dist_dated_avg = {} # apply averaging
        for date, dist in dist_dated.items():
            dist_dated_avg[date] = self.calc_avg_allmetrics(dist)


        # === extract distances on shuffled data ===

        #initialize container for null distribution
        #{date: {metric: null}}
        null = {}
        for date in dist_dated.keys():
            null[date] = {key: [] for key in self.metrics.keys()}

        for i in range(n_shuffles):
            self.shuffle('by_date')
            shuf_dist_dated = self.grab_dist('shuf_dated')

            for date, dist in shuf_dist_dated.items():
                shuf_dist_dated_avg = self.calc_avg_allmetrics(dist)

                for m_name, val in shuf_dist_dated_avg.items():
                    null[date][m_name].append(val)

        # calculate p-values
        pvals_by_date = {}
        for date, this_null in null.items():
            dist_avg = dist_dated_avg[date]
            pvals_by_date[date] = self.calc_p_allmetrics(this_null, dist_avg)

        # reformat to be {metric: {date : val}} instead of {date: {metric: val}}
        pvals_by_metric     = invert_nested_dict(pvals_by_date)  # p-values
        self.dist_dated_avg = invert_nested_dict(dist_dated_avg) # dists

        return pvals_by_metric, null



    def shuffle(self, mode):
        if mode=='all':
            # shuffle entire dataset, across sessions
            shuffle_indices = np.random.permutation(self.n)
            self.rel_poke_xy_shuf = self.rel_poke_xy[shuffle_indices]
        elif mode=='by_date':
            # shuffle within each date (session)
            for v in self.date_info.values():
                np.random.shuffle(v['new_ids'])
                self.rel_poke_xy_shuf[v['old_ids'], :] = self.rel_poke_xy[v['new_ids'], :]
        else:
            raise ValueError('Unknown shuffle mode ' + mode)


    def return_deepcopy(self):
        # return a deepcopy of this class instance
        return copy.deepcopy(self)



class sim_permuter(poke_permuter):
    def __init__(self, metrics, ntrials=100, datechunk_M=20, datechunk_SD=10, template=None):
        """
        to simulate single session, just set M = ntrials, SD = 0
        """

        xy_col_names = {'poke'  : 'poke_$',
                        'ref'   :  'ref_$',
                        'target':  'cue_$' }
        date_col = 'date'

        #=== create simulated data  ===
        sim = {'poke_x': np.zeros(ntrials), #poke
               'poke_y': np.zeros(ntrials)}


        if template is None:
            sim = {'cue_x' : np.zeros(ntrials), #target
                   'cue_y' : np.zeros(ntrials),
                   'ref_x' : np.zeros(ntrials), #ref
                   'ref_y' : np.zeros(ntrials)}
        elif template == 'random1': #ref, and rel_cue, uniform random in [-1,1]
            for ref in ['ref_x', 'ref_y']:
                sim[ref] = np.random.uniform(low=-1, high=1, size=ntrials)

            for cue, ref in zip(['cue_x', 'cue_y'], ['ref_x', 'ref_y']):
                sim[cue] = np.random.uniform(low=-1, high=1, size=ntrials) + sim[ref]
        else:
            raise ValueError('Template ' + template + 'does not exist!')

        sim_df = pd.DataFrame(sim)
        sim_df = self.generate_dates(sim_df, M=datechunk_M, SD=datechunk_SD)

        super().__init__(sim_df, xy_col_names, metrics, date_col)

        self.update_sim()

    def update_sim(self):
        A = np.array(self.df[['poke_x', 'poke_y']])
        B = np.array(self.df[['cue_x', 'cue_y']])
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


class metric_obj():
    """
    class that defines metric function
    implements 1. the metric distance, 2. the averager, 3. the direction of comparison with null hypothesis

    """
    def __init__(self, metric_name, averager_type, null_direction):

        self.metric_name = metric_name

        self.calc_dist = self.set_metric(metric_name)
        self.averager  = pyfun.stats.select_averager(averager_type)
        self.null_hyp  = self.set_null_hyp(null_direction)


    def set_metric(self, metric_name):
        self.metric_name = metric_name

        if metric_name   == 'euclid':
            func = pyfun.stats.euclid_distance
        elif metric_name == 'x_dist':
            func = pyfun.stats.distance_x
        elif metric_name == 'y_dist':
            func = pyfun.stats.distance_y
        elif metric_name == 'x_corr':
            func = pyfun.stats.corr_x
        elif metric_name == 'y_corr':
            func = pyfun.stats.corr_y
        else:
            raise ValueError('Invalid distance metric ' + str(metric_name))

        return func

    def set_null_hyp(self, null_direction):
        """
        Set null hypothesis direction, returning
        a function (shuf, orig) where orig is the null hypothesis value (typically a scalar),
        and shuf is an array of data to be compared.

        The function returns NaN if any NaN values are present in shuf or orig;
        otherwise, it performs the specified comparison across the array, returning a vector of True / Falses.
        The mean of this vector gives a p-value
        """
        if null_direction == 'less_than_or_equal':
            def comparison_function(shuf, orig):
                if np.isnan(orig).any() or np.isnan(shuf).any():
                    return [np.nan]
                else:
                    return shuf <= orig
        elif null_direction == 'greater_than':
            def comparison_function(shuf, orig):
                if np.isnan(orig).any() or np.isnan(shuf).any():
                    return [np.nan]
                else:
                    return shuf > orig
        else:
            raise ValueError('Unknown null direction specified.')

        return comparison_function


class perm_and_plotter():
    """
    usage: perm and plotter(plot_title, permer)

    given a permer (contains df and metrics),
    battery of shuffle tests across all metrics,
    and plotters.

    Plotting:
    - p-values for different metrics, across sessions, shared axes
    - actual value of each metric, separate twinned y-axis for each

    """

    def __init__(self, plot_title, permer):
        self.permer = permer
        self.plot_title = plot_title
        self.ax2_pos = {} #position of each additional twinned axis for each metric


    def format_plot(self):
        ax = self.ax
        ax.set_title(self.plot_title, ha='center', va='top')
        ax.spines[['right', 'top']].set_visible(False)
        #panplot.set_fonts()
        plt.rcParams['font.family'] = 'DejaVu Sans'

        ax.set_yticks([0, 0.5, 1])
        ax.set_ylim([-0.02, 1])

    def perm(self, n_shuffles):
        self.pvals_dated,  foo = self.permer.permtest_by_date(n_shuffles)
        self.pvals_fine,   foo = self.permer.permtest_all(n_shuffles, shuf_mode='by_date')
        self.pvals_coarse, foo = self.permer.permtest_all(n_shuffles, shuf_mode='all')

    def generate_plot(self, height=3, width_per_date=0.5, width_per_metric=0.5):
        """
        generate a new figure
        :return:
        """

        #calculate plot width
        width = 0
        n_sessions = len(self.permer.date_info)
        n_metrics  = len(self.permer.metrics) #more metric means need more space for twinned axes

        width += n_sessions * width_per_date
        width += n_metrics  * width_per_metric



        fig, ax = plt.subplots(figsize=(width, height))
        self.ax = ax
        self.plot()
        self.plot_annotate_overalls()

        return fig, ax

    def plot_annotate_overalls(self):
        """
        annotate overall p-vals
        :return:
        """

        # Text annotation below each twinned axis
        ax_position = self.ax.get_position()
        fig_width   = self.ax.figure.get_figwidth()
        dpi         = self.ax.figure.get_dpi()
        axis_width_in_points = ax_position.width * fig_width * dpi

        for metric_name, ax2_pos in self.ax2_pos.items():
            text_x = (ax_position.x0 + ax_position.width) + (ax2_pos / axis_width_in_points)

            pval = str(self.pvals_fine[metric_name])
            self.ax.text(text_x, -0.1, pval, transform=plt.gcf().transFigure,
                         ha='center', va='bottom', color=self.p_colors[metric_name])

    def plot(self):
        self.p_colors = panplot.generate_plot_colors(self.permer.metrics.keys())
        p_markers = panplot.generate_plot_markers(['p_val', 'dist'])

        ax2_pos = 10  # position of first secondary y-axis, be incremented with each metric

        for m in self.permer.metrics:
            pvals = self.pvals_dated
            ax = self.ax

            # === plots by date ===

            # primary plot axis for p values (one plot line per metric on shared y-axis)
            pval_plot = ax.plot(pvals[m].keys(), pvals[m].values(),
                                color=self.p_colors[m], label=m,
                                linewidth=1, marker=p_markers['p_val'])

            ax.axhline(0.05, color='k', linewidth=0.5)  # p = .05 line
            ax.legend()
            ax.tick_params(rotation=90)  # rotate dates for readability

            # plot metric values (e.g. distance, correlation...) with separate secondary y-axis
            # for each

            dist_m = self.permer.dist_dated_avg[m]
            ax2 = ax.twinx()
            dist_plot = ax2.plot(dist_m.keys(), dist_m.values(),
                                 color=self.p_colors[m], label=m,
                                 linestyle='--', marker=p_markers['dist'])

            ax2.spines['right'].set_position(('outward', ax2_pos))
            ax2.spines['right'].set_color(self.p_colors[m])  # Set the color of the spine to match the plot
            ax2.spines['right'].set_linestyle('--')
            ax2.tick_params(axis='y', colors=self.p_colors[m])  # Set the color of the ticks to match the plot

            self.ax2_pos[m] = ax2_pos
            ax2_pos += 50

        # add trial numbers per date
        for date, v in self.permer.date_info.items():
            ax.text(date, 0.05, v['n'], fontsize=14, color='r', alpha = 0.5, ha='center', va='center')

        # Create custom lines for the legend
        solid_line = mlines.Line2D([0, 1], [0, 1], color='black', linestyle='-',
                                   marker=p_markers['p_val'], label='p val',
                                   markevery=[-1])

        dashed_line = mlines.Line2D([0, 1], [1, 0], color='black', linestyle='--',
                                    marker=p_markers['dist'], label='avg',
                                    markevery=[-1])

        # Create a legend with these custom lines
        legend = ax.legend(handles=[solid_line, dashed_line], loc='upper center',
                           frameon=True, framealpha=0.6,
                           handlelength=2, bbox_to_anchor=(0.15, 1))

        ax.add_artist(legend)  # This ensures the first legend remains visible

        ax.legend()
        ax.set_ylabel('p-value')
        self.format_plot()