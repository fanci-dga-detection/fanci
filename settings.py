from enum import Enum

import pandas as pd
import logging
from datetime import datetime
import builtins
import os
from os.path import expanduser

from pathlib import Path

WORKSPACE_ROOT = expanduser("~") + '/dga_detection_workspace/'

PANDAS_PRINT_ALL_ROWS_AND_COLUMNS = True

LOG_TO_FILE = True
LOG_LVL = logging.INFO

LOG_ROOT = WORKSPACE_ROOT + 'log/'
PCAPS_FOLDER = WORKSPACE_ROOT + 'pcaps/'
DGARCHIVE_FODLER = WORKSPACE_ROOT + 'dgarchive/'
SETS_FOLDER = WORKSPACE_ROOT + 'sets/'
GRID_SEARCH_FOLDER = WORKSPACE_ROOT + 'grid_search/'
GRID_SEARCH_FOLDER_TMP = WORKSPACE_ROOT + 'grid_search/tmp/'
ANALYSIS_FOLDER = WORKSPACE_ROOT + 'analysis/'
EVAL_FOLDER = WORKSPACE_ROOT + 'eval/'
CLF_FOLDER = WORKSPACE_ROOT + 'clfs/'
WORKSPACE_FILE = WORKSPACE_ROOT + 'workspace.pkl'
PCAP_CLASSIFICATION_FOLDER = ANALYSIS_FOLDER + 'pcap_classification_results_mt_data/'


class SetTypes(Enum):
    single_dga = 'single_dga'
    single_dga_grouped_seed = 'single_dga_grouped_seed'
    single_dga_grouped_day = 'single_dga_grouped_day'
    mixed_dga_grouped_family = 'mixed_dga_grouped_family'


####################################################
#  NO MODIFICATION BEYOND THIS DELIMITER NECESSARY #
####################################################

LOGGER_CONFIGURED = False
PRINT_TO_LOG_CONVERT = False
SCRIPTS_ROOT, _ = os.path.split(os.path.realpath(__file__))
NOW_STR = datetime.now().strftime('%Y_%m_%d_%H_%M_')

if not os.path.exists(PCAPS_FOLDER) and not Path(PCAPS_FOLDER).exists() and not Path(PCAPS_FOLDER).is_symlink():
    os.makedirs(PCAPS_FOLDER)
if not os.path.exists(SETS_FOLDER):
    os.makedirs(SETS_FOLDER)
if not os.path.exists(DGARCHIVE_FODLER):
    os.makedirs(DGARCHIVE_FODLER)
if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT)
if not os.path.exists(GRID_SEARCH_FOLDER):
    os.makedirs(GRID_SEARCH_FOLDER)
if not os.path.exists(GRID_SEARCH_FOLDER_TMP):
    os.makedirs(GRID_SEARCH_FOLDER_TMP)
if not os.path.exists(ANALYSIS_FOLDER):
    os.makedirs(ANALYSIS_FOLDER)
if not os.path.exists(EVAL_FOLDER):
    os.makedirs(EVAL_FOLDER)
if not os.path.exists(CLF_FOLDER):
    os.makedirs(CLF_FOLDER)
if not os.path.exists(PCAP_CLASSIFICATION_FOLDER):
    os.makedirs(PCAP_CLASSIFICATION_FOLDER)

for s in SetTypes:
    for f in [SETS_FOLDER, EVAL_FOLDER]:
        if not os.path.exists(f + s.value):
            os.makedirs(f + s.value)

log = None


def configure_pandas_print():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)


if PANDAS_PRINT_ALL_ROWS_AND_COLUMNS:
    configure_pandas_print()


def configure_logger():
    global LOGGER_CONFIGURED, log
    if not LOGGER_CONFIGURED:

        logging.Logger.manager.loggerDict.clear()
        logging.VERBOSE = 5
        logging.addLevelName(logging.VERBOSE, 'VERBOSE')
        logging.Logger.verbose = lambda inst, msg, *args, **kwargs: inst.log(logging.VERBOSE, msg, *args, **kwargs)
        logging.verbose = lambda msg, *args, **kwargs: logging.log(logging.VERBOSE, msg, *args, **kwargs)

        log = logging.getLogger('log')
        log.setLevel(LOG_LVL)
        log_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

        if LOG_TO_FILE:
            file_handler = logging.FileHandler(datetime.now().strftime(LOG_ROOT + 'learning_%Y_%m_%d_%H_%M_.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(log_formatter)
            log.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        log.addHandler(console_handler)
        if PRINT_TO_LOG_CONVERT:
            builtins.print = log_print
        LOGGER_CONFIGURED = True


def log_print(*values, sep: str = ' ', end: str = '', file=None, flush: bool = False):
    try:
        if len(values) > 0:
            if values[0].strip():
                log.info(values[0].strip())
    except:
        log.error('Print redirect failure')

configure_logger()
COPY_SUFFIX = '_copy_'
SCORE_SUFFIX = '_scoring'

LOCAL_FILTER_SET_RWTH = ['rwth-aachen.de', 'sophosxl.net', 'in-addr.arpa', 'fh-aachen.de', 'ip6.arpa', 'fritz.box',
                    'mcafee.com', 'isc.org', 'senderbase.org', 'epic.com', 'spamhaus.org', 'netgear.com',
                    'spameatingmonkey.net', 'aachen.de', 'weh.ac', 'mshome.net', 'spamcop.net', 'airtelbroadband.in']

LOCAL_FILTER_SET_SIEMENS = ['sophosxl.net', 'in-addr.arpa', 'ip6.arpa', 'fritz.box', 'mcafee.com', 'airtelbroadband.in',
                            'trendmicro.com', 'bayer.com', 'sts.net', 'usssl01.net', 'my-it-solutions.net', 'performax.cz', 'it-solutions.net',
                            'ohenc.net', 'ibs-ag.us', 'audiology-solutions.net', 'netsiemens.net', 'vodafone.in', 'amazonaws.com', 'siemsn.net', 'siemens.net',
                            't-ipconnect.de', 'siemens.org']