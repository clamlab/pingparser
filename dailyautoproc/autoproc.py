import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import yaml
import argparse #for handling command-line args
import importlib
from datetime import datetime

import pingparser.general as genparse
import pingparser.scrapers as scrapers
import pingparser.containers as containers
import pyfun.bamboo as boo
import pyfun.timestrings as timestr
import pingparser.check as check

# Load YAML config
def setup_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    config = setup_directories(config)

    config['raw_dir'] = os.path.join(config['root_dir'], 'raw')

    config['proc_list_fn'] = os.path.join(config['dirs']['metadata'], 'processed.csv')
    #list containing already-pre-processed results files

    return config

def setup_directories(config):

    base_dir = os.path.join(config['root_dir'], 'processed', config['version'])

    # === get sub_dir paths ===
    # for extractors (e.g., events, tracking)
    sub_dirs = {
        f"{e_name}_dir": os.path.join(base_dir, e_name)
        for e_name in config['extractors']
    }

    # for metadata
    sub_dirs['metadata'] = os.path.join(base_dir, 'metadata')
    sub_dirs['logs']     = os.path.join(base_dir, 'metadata/logs')

    config['dirs'] = sub_dirs

    if os.path.exists(base_dir):
        # all the required subdirectories should already exist if the base_dir exists
        pass
    else:
        # otherwise, create the subdirectories
        for sub_dir in sub_dirs.values():
            os.makedirs(sub_dir,exist_ok=True)

    return config



# Set up logging based on config
def setup_logging(log_dir, debug):
    # Create a timestamp for the log file (e.g., '2024-10-22_14-30-00')
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Create the log filename with the timestamp
    fn = f'{timestamp}.log'

    if debug: #if running with --debug flag, amend log file name
        fn = f'debug-{fn}'

    log_file = os.path.join(log_dir, fn)

    # Set up logging configuration
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
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
            # Try to dynamically import the module
            extractors[e_name] = importlib.import_module(module_path)

            # Logging successful import
            logging.info(f'{e_name}: {module_path}')

        except ModuleNotFoundError as e:
            # Log error if the module cannot be found
            logging.error(f"ModuleNotFoundError: Failed to import {module_path}. Error: {e}")

        except ImportError as e:
            # Log other import-related errors
            logging.error(f"ImportError: Failed to import {module_path}. Error: {e}")

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

def load_processed_list(config):

    if not os.path.exists(config['proc_list_fn']):
        logging.info(f"Processed list not found. Creating {config['proc_list_fn']}")

    proc_list = boo.read_csv_or_create(config['proc_list_fn'], colnames=['anim', 'sess', 'length'])

    return proc_list

def preprocess(expt, extractors, cutoff_min, proc_list, overwrite):
    rows_to_delete = []

    #calculate total work to be done (roughly--this ignores how much is already processed)
    total_work = sum(len(anim.subsess_paths) for anim in expt.anim.values())

    with tqdm(total=total_work, desc="Preprocessing Sessions") as pbar:
        for anim_name, anim in expt.anim.items():
            anim_proc_list = boo.slice(proc_list, {'anim': [anim_name]})['sess']
            this_cutoff_min = cutoff_min.get(anim_name, pd.to_datetime(cutoff_min['default'])) #return second value if anim_name not in dict

            subsess = {}
            new_proc_list, new_proc_len = [], []


            for subsess_name, fn in anim.subsess_paths.items():
                pbar.update(1)

                overwrite_msg = ''
                if subsess_name in anim_proc_list.values:  # already processed
                    if overwrite:
                        row_num = anim_proc_list[anim_proc_list == subsess_name].index[0]
                        rows_to_delete.append(row_num)
                        logging.warning(f'Overwriting {subsess_name}')
                    else:
                        continue

                if timestr.search(subsess_name)[1] < this_cutoff_min:
                    continue
                df_sess_raw = genparse.read_raw(fn['Events'])
                df_sess = genparse.sess_summary(df_sess_raw, subsess_name, extractors['events'])
                subsess[subsess_name] = df_sess

                new_proc_list.append(subsess_name)
                new_proc_len.append(len(df_sess))

            new_df = pd.DataFrame({'anim': anim_name, 'sess': new_proc_list, 'length': new_proc_len})
            proc_list = pd.concat([proc_list, new_df], ignore_index=True)

            anim.subsess = subsess


    proc_list = proc_list[~proc_list.index.isin(rows_to_delete)]
    proc_list['length'] = proc_list['length'].astype('int')
    proc_list = proc_list.sort_values(['anim', 'sess'], ignore_index=True)
    return proc_list, expt

def save_processed_list(proc_list, proc_list_fn):
    proc_list.to_csv(proc_list_fn, index=False)
    logging.info("Updated processed list.")

def merge_sessions(expt):
    # === merge all new session dataframes occurring on the same day ===
    # TODO: what happens if session is over midnight?
    for anim_name, anim in expt.anim.items():
        # convert keys (in time string format) to datetime objects
        subsess_dt = {timestr.search(key)[1]: value for key, value in anim.subsess.items()}
        anim.sess = boo.merge_within_day(subsess_dt)


    # === merge all new sessions into single df===
    for anim_name, anim in expt.anim.items():
        # Concatenate the dataframes in the correct order
        anim.new_df = boo.concat_df_dicts(anim.sess, reset_index=True)



    # === perform basic checks of df lengths after merges ===

    #foodate = '2024-08-18T11_20_00'
    #expt.anim['EC01'].subsess[foodate] = expt.anim['EC01'].subsess[foodate].iloc[0:10]

    err_msgs = check.subsess_lens(expt)  # sanity check
    if err_msgs:
        prefix = 'Subsess merge length: '
        for err_msg in err_msgs:
            logging.error(prefix + err_msg)

    err_msgs = check.new_df_lens(expt)  # sanity check
    if err_msgs:
        prefix = 'Omnibus merge length: '
        for err_msg in err_msgs:
            logging.error(prefix + err_msg)

    return expt

def save_events(expt, output_root):

    for anim_name, anim in expt.anim.items():
        if len(anim.new_df) == 0:
            logging.info(f"{anim_name}: no new sessions found.")
            continue
        fn = os.path.join(output_root, anim_name + '.csv')

        new_sess = anim.new_df.sess.unique()
        if os.path.exists(fn):
            big_df = pd.read_csv(fn)
        else:
            anim.new_df.to_csv(fn, index=False)
            logging.info(f"{fn} doesn't exist. Creating...")
            continue

        # check proportion of big_df trials with sessions that are in new_df
        # if all is done right, it should be 0 for non-overwriting case
        n_old = np.sum(big_df.sess.isin(new_sess))
        p_old = n_old / len(big_df) if len(big_df) > 0 else 0
        if p_old > 0:
            logging.info(f"{anim_name}: {n_old} trials ({p_old * 100:.2f}%) in big_df are contained in new_df sessions.")
            user_input = input("Overwrite? (y/n)")
            if user_input == 'y':
                big_df = boo.slice(big_df, {'sess': new_sess}, '-')  # remove old data
        else:
            logging.info(f"{anim_name}: adding new sessions")

        big_df = pd.concat([big_df, anim.new_df])
        big_df = big_df.sort_values(by=['sess', 'TrialNum'], ascending=[True, True])
        big_df.to_csv(fn, index=False)
        anim.big_df = big_df
        logging.info(f"Saved new sessions for {anim_name}.")

def extract_and_save_touch(expt, extractor, config):
    tracker_name = 'Touch_xy_bs'

    output_root = config['dirs'][f"{tracker_name}_dir"]

    for anim_name, anim in expt.anim.items():
        print(anim_name)
        fd = os.path.join(output_root, anim_name)

        # create rat folder if it no exist
        if not os.path.exists(fd):
            os.makedirs(fd)

        for subsess_name in anim.subsess:
            subsess_path = anim.subsess_paths[subsess_name]
            if tracker_name not in subsess_path:
                continue

            fn = os.path.join(fd, subsess_name + '.csv')

            if os.path.exists(fn):
                if config['overwrite']:
                    to_write = 'y'
                else:
                    to_write = 'n'
            else:
                to_write = 'y'

            if to_write == 'y':
                xy_df = extractor.resp_xy(subsess_path[tracker_name], subsess_name, tracker_name)
                xy_df.to_csv(fn, index=False)



def main(config_file, debug=False):
    try:
        # Load config file and add paths
        config = setup_config(config_file)

        # start logger
        setup_logging(config['dirs']['logs'], debug)

        # Initialize containers for animal data
        expt = init_containers(config)

        # Load extractor modules
        extractors = load_extractors(config)

        # Load raw file paths
        load_file_paths(expt)

        # Load list of already pre-processed sessions
        proc_list = load_processed_list(config)

        # Preprocess sessions, skipping those already in list, and update list
        proc_list, expt = preprocess(expt, extractors, config['cutoff_min'], proc_list, config['overwrite'])

        # Merge sessions occurring on same day
        expt = merge_sessions(expt)

        #need some way to deal with contingencies e.g. what if program is interrupted in any of the next 3 lines??
        #extract touch
        extract_and_save_touch(expt, extractors['Touch_xy_bs'], config)

        # Save events
        save_events(expt, config['dirs']['events_dir'])


        # Save processed list
        save_processed_list(proc_list, config['proc_list_fn'])



        logging.info("Processing completed successfully.")

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}", exc_info=True)


if __name__ == "__main__":
    # Example usage: python autoproc.py --config configs/config_v05.yaml

    # Default to v05 config file for debugging purposes
    default_config_path = 'configs/config_v06.yaml'

    parser = argparse.ArgumentParser(description='Run auto processing with configuration.')
    parser.add_argument('--config', type=str, default=default_config_path, help='Path to the config file')

    # Optional --debug flag to enable debug mode
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the specified config (defaulting to v05)
    main(args.config, debug=args.debug)
