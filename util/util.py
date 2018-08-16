import pickle

import logging

import settings
from data_processing import data

log = logging.getLogger('log')

def open_grid_search_file(file: str):
    """
    Open a grid search pickle file to be able to view results again.
    :param file: 
    :return: 
    """
    with open(file, 'rb') as f:
        grid_file = pickle.load(f)
    return grid_file

# TODO grid file cummulation
# TODO grid search "splitting"
# TODO useful parameters for grid search
# TODO!!! leave out DGA kfold validiation !!! (+ leave out seed)

# TODO plotting results of grid search? KFold validation??


def unique_domains_cleaner(file):
    tmp = set()
    with open(file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            tmp.add(line.strip().lower())

    log.info('Set size before cleaning {!s}'.format(len(tmp)))

    tmp = set(data.clean_domain_list(tmp))

    log.info('Cleaning Done. STarting TLD filtering')

    tmp2 = data.filter_non_valid(tmp)

    log.info('TLD Filtering done, starting diff.')

    dif = [item for item in tmp if item not in tmp2]

    log.info('Set size after cleaning {!s}'.format(len(tmp2)))

    log.info('Set size before cleaning {!s}'.format(dif))

    del dif
    del tmp

    data.write_unique_set_to_file(tmp2, settings.WORKSPACE_ROOT + 'unique_domains_cache_cleaned_no_tld.txt')
