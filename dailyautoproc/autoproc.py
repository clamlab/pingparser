import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import yaml
import argparse #for handling command-line args
import importlib

from pyfun.singleton import SingletonInstance
import atexit

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


from datetime import datetime

import pingparser.general as genparse
import pingparser.scrapers as scrapers
import pingparser.containers as containers
import pyfun.bamboo as boo
import pyfun.timestrings as timestr
import pingparser.check as check

import glob
import time

# Load YAML config
def setup_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    config = setup_paths(config)

    return config

def setup_paths(config):
    # TODO in theory dir creation only needs to be run first time. Unclear what is best practice for
    # TODO checking this.


    # === common subdirectories ===
    config['raw_dir']    = os.path.join(config['root_dir'], 'raw')
    config['output_dir'] = os.path.join(config['root_dir'], 'processed')

    config['log_dir'] = os.path.join(config['output_dir'], 'logs')

    os.makedirs(config['log_dir'], exist_ok=True) #path for log_dir


    #=== subdirectories for separate extractors ===
    extractor_output_paths = {}

    # Create versioned subdirectories for each extractor based on module name
    for e_name, e_module in config['extractors'].items():
        versioned_dir = os.path.join(config['output_dir'], e_name, e_module)

        # Base paths
        paths = {
            'data': os.path.join(versioned_dir, 'data'),
            'metadata': os.path.join(versioned_dir, 'metadata'),
        }

        # Add 'big_df' path only for 'Events' #we don't really need omnibus table for trackers / file too big
        if e_name == 'Events':
            for s in ['big_df', 'session_stats']:
                paths[s] = os.path.join(versioned_dir, s)


        # Assign to the extractor output paths
        extractor_output_paths[e_name] = {
            'dirs': paths,
            'proc_list_fn': os.path.join(versioned_dir, 'metadata', 'processed.csv'),
        }

    # Create all directories in each entry
    for paths_dict in extractor_output_paths.values():
        for dir_path in paths_dict['dirs'].values():
            os.makedirs(dir_path, exist_ok=True)

    config['paths'] = extractor_output_paths
    return config




# Set up logging based on config
def setup_logging(config, debug):
    # Create a timestamp for the log file (e.g., '2024-10-22_14-30-00')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create the log filename with the timestamp
    fn = f'{timestamp}.log'

    if debug: #if running with --debug flag, amend log file name
        fn = f'debug-{fn}'

    log_file = os.path.join(config['log_dir'], fn)

    # Set up logging configuration
    logging.addLevelName(logging.WARNING, "WARN") #shortern "WARNING" in log file to "WARN"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"  # Only show hours, minutes, seconds
    )


def init_containers(config):
    expt = containers.Experiment()
    expt.data_root = config['raw_dir']
    for a in config['animal_names']:
        expt.anim[a] = containers.AnimalData(a)

    return expt

def load_extractors(config):
    extractors = {} #container for extractor modules
    for e_name, e_ver in config['extractors'].items():

        module_path = f"pingparser.extractors.{e_name}.{e_ver}"
        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)

            # Instantiate the Extractor class within the imported module
            extractor_class = getattr(module, "Extractor")  # Assuming each module has an Extractor class
            extractors[e_name] = extractor_class()  # Instantiate the class

            # Logging successful import
            logging.info(f'{e_name}: {module_path}')

        except ModuleNotFoundError as e:
            # Log error if the module cannot be found
            logging.error(f"ModuleNotFoundError: Failed to import {module_path}. Error: {e}")

        except ImportError as e:
            # Log other import-related errors
            logging.error(f"ImportError: Failed to import {module_path}. Error: {e}")

        except AttributeError as e:
            # Handle case where the module does not have an Extractor class
            logging.error(f"AttributeError: {module_path} does not contain an 'Extractor' class. Error: {e}")

    return extractors

def load_file_paths(expt):
    for anim_name, anim in expt.anim.items():
        anim_root = os.path.join(expt.data_root, anim_name)
        anim.subsess_paths, timestamp_errors = scrapers.get_subsess_paths(anim_name, anim_root, verbose=True)

        if len(timestamp_errors) > 0:
            for error in timestamp_errors:
                # Format each error (each error is a list with two elements)
                formatted_error = f"Path: {error[0]}, File: {error[1]}"
                logging.error(formatted_error)

    return expt

def load_processed_list(config):
    proc_list = {}
    for e_name in config['extractors']:
        fn = config['paths'][e_name]['proc_list_fn']

        if not os.path.exists(fn):
            logging.info(f"Processed list not found. Creating {fn}")

        proc_list[e_name] = boo.read_csv_or_create(fn, colnames=['anim', 'sess', 'length'])

    return proc_list

def save_session_data(session_file_path, df_sess):
    """Save session data to CSV file."""
    os.makedirs(os.path.dirname(session_file_path), exist_ok=True)
    df_sess.to_csv(session_file_path, index=False)
    logging.info(f"Saved: {session_file_path}")

def save_session_stats(session_stats_df, fn):
    # Check if the file exists
    if os.path.exists(fn):
        # Load existing data into old_session_stats_df
        old_session_stats_df = pd.read_csv(fn)

        # Concatenate the new and old DataFrames
        combined_df = pd.concat([old_session_stats_df, session_stats_df])

        # Drop duplicate rows based on the 'date' column, keeping the last occurrence
        updated_df = combined_df.drop_duplicates(subset='date', keep='last')
    else:
        # If the file does not exist, use the current DataFrame
        updated_df = session_stats_df

    # Save the updated DataFrame to a temporary file first
    temp_file = f"{fn}.temp"
    updated_df.to_csv(temp_file, index=False)

    # Atomically replace the old file with the updated file
    os.replace(temp_file, fn)
    logging.info(f"Saved session stats to {fn}.")



def extract_session_data(extractor, df_sess_raw, subsess_name):
    """Extract session data using the provided extractor."""
    logging.info(f"Extracting: {subsess_name}")
    return extractor.extract(df_sess_raw, subsess_name)

def preprocess(config, expt, extractor, proc_list, batch_save_interval=10, debug=False):
    """
    Preprocess and save each session, updating proc_list incrementally to ensure robustness.
    """
    proc_list_path = config['paths'][extractor.TYPE]['proc_list_fn']  # Path to proc_list CSV file for this extractor
    data_dir = config['paths'][extractor.TYPE]['dirs']['data']  # Directory for session data

    # Calculate the total work--quick and coarse way--doesn't pre-check if session already processed etc.
    total_work = sum(len(anim.subsess_paths) for anim in expt.anim.values())

    with tqdm(total=total_work, desc=f"Preprocessing {extractor.TYPE}") as pbar, \
         ProcessPoolExecutor(max_workers=4) as extract_executor, \
         ThreadPoolExecutor(max_workers=4) as save_executor:

        for anim_name, anim in expt.anim.items():
            data_dir_anim = os.path.join(data_dir, anim_name)
            # Filter proc_list for this animal's sessions
            anim_proc_list = boo.slice(proc_list, {'anim': [anim_name]})['sess']

            # cross check list of processed files against number of files actually existing.
            count_existing_files(anim_proc_list, data_dir_anim)


            this_cutoff_min = config['cutoff_min'].get(anim_name, pd.to_datetime(config['cutoff_min']['default']))

            subsess = {}
            subsess_stats = {}

            new_entries = []  # Temporary list to hold new entries for proc_list

            session_futures = {}  # futures and their session names (for saving)

            for subsess_name, fn in anim.subsess_paths.items():

                # conditions for skipping
                should_skip = (
                        subsess_name in anim_proc_list.values or  # Already processed
                        timestr.search(subsess_name)[1] < this_cutoff_min or  # Before cutoff date
                        extractor.TYPE not in fn  # Missing required data file--could happen if program is interrupted early
                )

                if should_skip:
                    pbar.update(1)
                    continue



                # Define the full path for the session file in the correct directory
                session_file_path = os.path.join(data_dir_anim, f"{subsess_name}.csv")


                #submit the job, and the "ticket" is stored as a "future"
                future = extract_executor.submit(extract_session_data, extractor,
                                                 fn[extractor.TYPE], subsess_name)

                #store each future as a session key, and then associated info as values
                session_futures[future] = (anim_name, subsess_name, session_file_path)


            # === save each session (unmerged) separately as they get processed ===
            for future in as_completed(session_futures):
                anim_name, subsess_name, session_file_path = session_futures[future]

                # Retrieve the extracted data
                df_sess, df_sess_stats = future.result()
                subsess[subsess_name] = df_sess

                subsess_stats[subsess_name] = df_sess_stats
                # Schedule the save operation concurrently using ThreadPoolExecutor (for I/O tasks)
                save_executor.submit(save_session_data, session_file_path, df_sess)

                # Update proc_list with the new entry
                new_entries.append({"anim": anim_name, "sess": subsess_name, "length": len(df_sess)})
                pbar.update(1)
            pbar.refresh()


            # === ONLY FOR EVENTS: merge (sameday), then save, session stats ===
            try:
                merged_stats = merge_sameday_stats(subsess_stats)
            except:
                pass
            # Convert dictionary to DataFrame
            session_stats_df = pd.DataFrame.from_dict(merged_stats, orient='index').reset_index()
            session_stats_df.rename(columns={'index': 'date'}, inplace=True)
            fn = os.path.join(config['paths']['Events']['dirs']['session_stats'], f"{anim_name}.csv")
            save_session_stats(session_stats_df, fn)

            # Append new entries to proc_list and save incrementally after each animal
            if new_entries:
                new_df = pd.DataFrame(new_entries)
                proc_list = pd.concat([proc_list, new_df], ignore_index=True)

                proc_list = proc_list.sort_values(['anim', 'sess'], ignore_index=True) #sort

                # Save proc_list to a temporary file first, then rename it
                temp_file = f"{proc_list_path}.temp"
                proc_list.to_csv(temp_file, index=False)
                os.replace(temp_file, proc_list_path)  # Atomically replace the old file
                logging.info(f"Added new {anim_name} entries to proc_list.")

            anim.subsess = subsess  # Store processed data for the animal

    #check that processed list count matches existing files in directory
    check_proc_list(proc_list, data_dir, config)

    return proc_list, expt


def check_proc_list(proc_list, data_dir, config):
    """
    Checks the integrity of the proc_list:
    1. Verifies that the number of entries in proc_list equals the number of unique rows defined by 'anim' and 'sess'.
    2. Verifies that the total number of entries in proc_list equals the total number of .csv files across all
       subfolders in data_dir corresponding to animal names in config['animal_names'].

    Parameters:
    - proc_list (pd.DataFrame): The processed list DataFrame containing 'anim' and 'sess' columns.
    - data_dir (str): The root directory containing data subfolders.
    - config (dict): Configuration dictionary containing 'animal_names'.

    Returns:
    - dict: A dictionary summarizing the checks and any discrepancies found.
    """


    # Check 1: Unique entries in proc_list
    unique_count = proc_list[['anim', 'sess']].drop_duplicates().shape[0]
    proc_list_length = len(proc_list)

    unique_check_passed = unique_count == proc_list_length
    unique_check_message = (
        f"Check duplicates in proc list--{'passed' if unique_check_passed else 'failed'}: "
        f"{unique_count} unique, {proc_list_length} total."
    )
    if unique_check_passed:
        logging.info(unique_check_message)
    else:
        logging.warning(unique_check_message)

    # Check 2: Total number of files across subfolders

    total_file_count = 0
    for anim_name in config['animal_names']:
        anim_folder = os.path.join(data_dir, anim_name)
        if os.path.exists(anim_folder):
            total_file_count += sum(1 for f in os.listdir(anim_folder) if f.endswith('.csv'))

    file_check_passed = total_file_count == proc_list_length
    file_check_message = (
        f"Check file count vs proc list--{'passed' if file_check_passed else 'failed'}: "
        f"{total_file_count} found,  {proc_list_length} in list."
    )


    if file_check_passed:
        logging.info(file_check_message)
    else:
        logging.warning(file_check_message)



def count_existing_files(proc_list, data_dir):
    """
    Cross-checks the number of processed sessions in `proc_list` with the number of files in `data_dir`.
    Logs a warning if there is a discrepancy.
    """
    # Count unique sessions in proc_list for the extractor
    unique_sessions_count = len(proc_list)

    # Count the number of .csv files in the data directory (for the extractor)
    if not os.path.exists(data_dir):
        saved_files_count = 0
    else:
        saved_files_count = sum(1 for f in os.listdir(data_dir) if f.endswith('.csv'))

    # Log the results of the check
    if unique_sessions_count != saved_files_count:
        logging.warning(
            f"Mismatch: {unique_sessions_count} sessions in proc_list but {saved_files_count} files in {data_dir}.")




def merge_sameday_sessions(dfs_by_timestr):
    # === merge all new session dataframes occurring on the same day ===
    # TODO: what happens if session is over midnight?

    # convert keys (in time string format) to datetime objects
    dfs_by_time = {timestr.search(key)[1]: value for key, value in dfs_by_timestr.items()}

    merged_sessions = boo.merge_within_day(dfs_by_time)

    return merged_sessions

def merge_sameday_stats(subsess_stats):
    # convert keys (in time string format) to datetime objects
    subsess_stats = {timestr.search(key)[1]: value for key, value in subsess_stats.items()}
    grouped_dates = boo.group_datetime_objects_by_date(subsess_stats.keys())

    merged_stats = {}
    for date, sess_times in grouped_dates.items():
        merged_dict = {}
        this_merged_stats = {}

        for s in sess_times:
            this_sess_stats = subsess_stats[s]
            for stat_name, stat_value in this_sess_stats.items():
                if stat_name not in merged_dict:
                    merged_dict[stat_name] = []

                merged_dict[stat_name].append(stat_value)

        if len(merged_dict) == 0:
            continue

        earliest =  min(merged_dict['start_time'])
        latest   =  max(merged_dict['end_time'])
        latest_index = merged_dict['end_time'].index(latest) #use for getting latest trialtype on each day

        this_merged_stats.update({'start_time': earliest.strftime('%H:%M'),
                                  'end_time'  :   latest.strftime('%H:%M')})

        this_merged_stats['last_TrialType'] = merged_dict['last_TrialType'][latest_index]

        for stat_name in ['n_correct', 'n_attempted', 'n_total']:
            this_merged_stats[stat_name] = np.sum(merged_dict[stat_name])

        merged_stats[date] = this_merged_stats

    merged_stats = {key.strftime('%Y-%m-%d'): value for key, value in merged_stats.items()} #convert keys back to str

    return merged_stats


def load_file_for_omnibus(file_path):
    """Loads a file and returns its timestamp and DataFrame."""
    datetime_str = os.path.basename(file_path).split('.')[0]
    try:
        df = pd.read_csv(file_path)
        return datetime_str, df
    except pd.errors.EmptyDataError:
        return datetime_str, None

def save_col_types(config, extractors):
    # save column dtypes of extractor to metadata directory
    # useful when loading dfs later: newer pandas encourage / force explicit dtypes

    for e_name, e in extractors.items():
        fn = os.path.join(config['paths'][e_name]['dirs']['metadata'],
                          'coltypes.csv')

        if os.path.exists(fn):
            #file already created (should be done on first run)
            continue
        else:
            logging.info(f"Creating {fn}")
            pd.DataFrame(list(e.COLUMN_DTYPES.items()),
                         columns=['colname','dtype']).to_csv(fn, index=False)


# Merge all sessions to form omnibus df and save in parallel
def create_omnibus(anim_name, anim, config, e_name):
    # Concatenate the dataframes in the correct order
    big_df = boo.concat_df_dicts(anim.sess, reset_index=True)
    fn = os.path.join(config['paths'][e_name]['dirs']['big_df'], anim_name + '.csv')
    big_df.to_csv(fn, index=False)
    return anim_name, fn, big_df


def main(config_file, debug=False):
    try:
        # Load config file and add paths
        config = setup_config(config_file)

        # start logger
        setup_logging(config, debug)

        # Initialize containers for animal data
        expt = init_containers(config)

        # Load extractor modules
        extractors = load_extractors(config)

        # Load raw file paths, store in anim.subsess_paths for each animal
        expt = load_file_paths(expt)

        # Load list of already pre-processed sessions
        proc_list = load_processed_list(config)

        # save metadata
        save_col_types(config, extractors)



        # Preprocess sessions, skipping those already in list, and update list
        for e_name in config['extractors']:
            logging.info(f"Processing {e_name}")
            updated_proc_list, expt = preprocess(config, expt,
                                                 extractors[e_name],
                                                 proc_list[e_name], debug=debug)

            proc_list[e_name] = updated_proc_list #this has already been saved in preprocess()


        # pull other metadata e.g. start and end time of session (across sub-sessions)
        # TODO: refactor as extractor? (but we don't need to save subsession info and may be overkill because the data format may be too sparse)


        # reconstruct each subject's overall Events table. hacky for now,
        e_name = 'Events' #possible to loop through a list of extractor names, but we only want omnibus for events
        data_dir = config['paths'][e_name]['dirs']['data']

        total_work = len(expt.anim)

        with tqdm(total=total_work, desc=f"Pulling {e_name} sessions") as pbar:
            for anim_name in config['animal_names']:
                anim_folder = os.path.join(data_dir, anim_name)
                anim_files = glob.glob(os.path.join(anim_folder, "*.csv"))

                dfs_by_timestr = {}

                with ThreadPoolExecutor(max_workers=16) as executor: #parallel processing
                    future_to_file = {executor.submit(load_file_for_omnibus, f): f for f in anim_files}

                    for future in as_completed(future_to_file):
                        datetime_str, df = future.result()
                        if df is not None:
                            dfs_by_timestr[datetime_str] = df

                merged_sessions = merge_sameday_sessions(dfs_by_timestr)

                expt.anim[anim_name].subsess = dfs_by_timestr
                expt.anim[anim_name].sess    = merged_sessions
                pbar.update(1)


        with tqdm(total=total_work, desc=f"Saving {e_name} omnibus") as pbar:
            with ThreadPoolExecutor(max_workers=16) as executor:
                # Prepare the futures for parallel execution
                futures = {
                    executor.submit(create_omnibus, anim_name, anim, config, e_name): anim_name
                    for anim_name, anim in expt.anim.items()
                }

                # Process the results as they complete
                for future in as_completed(futures):
                    anim_name = futures[future]  # Retrieve the associated animal name
                    try:
                        anim_name, file_path, big_df = future.result()
                        expt.anim[anim_name].big_df = big_df
                        logging.info(f"Saved omnibus dataframe for {anim_name} to {file_path}")
                    except Exception as e:
                        logging.error(f"Failed to save dataframe for {anim_name}. Error: {e}")

                    pbar.update(1)

        pbar.refresh()

        err_msgs = check.merged_df_lens(expt, verbose=False)  # sanity check
        if err_msgs:
            prefix = 'Omnibus merge length: '
            for err_msg in err_msgs:
                logging.error(prefix + err_msg)
        else:
            logging.info(f"Omnibus merge length checks passed: {e_name}")


        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}", exc_info=True)


if __name__ == "__main__":
    # Example usage: python autoproc.py --config configs/config_v05.yaml

    script_name = os.path.basename(__file__)

    # === ensure that only one instance of this script is running at any time, by using a lock file ===

    # Define an absolute path for the lock file
    lock_dir = "C:\\script_locks"
    os.makedirs(lock_dir, exist_ok=True)  # Ensure the directory exists
    lock_file = os.path.join(lock_dir, "autoproc_py.lock")  # Unique lock file for your script

    singleton = SingletonInstance(lock_file, script_name)
    singleton.acquire_lock()
    # Ensure the lock is released when the script exits
    atexit.register(singleton.release_lock)

    # ==================================================================================================


    # Default config file for debugging
    default_config_path = 'configs/config_v08.yaml'

    parser = argparse.ArgumentParser(description='Run auto processing with configuration.')
    parser.add_argument('--config', type=str, default=default_config_path, help='Path to the config file')

    # Optional --debug flag to enable debug mode
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the specified config (defaulting to v05)
    main(args.config, debug=args.debug)
