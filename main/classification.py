import logging
import random
import string
from os import walk
from itertools import islice

import os

from sklearn.externals import joblib
from sklearn.externals.joblib.parallel import Parallel, delayed

import settings
import ntpath

from data_processing import data
from data_processing.data import PublicSuffixes
from data_processing.pcap_domain_utils import pcap_iterator
from learning.classifiers import Clf

log = logging.getLogger('log')


def classify_pcaps(path_to_pcaps_folder, clf: Clf, n_jobs=2):
    pcaps = []
    for (path, dirs, files) in walk(path_to_pcaps_folder):
        pcaps.extend(files)
        break

    pcaps = [path_to_pcaps_folder + '/' + pcap for pcap in pcaps if not pcap.endswith('.txt')]

    parallel = Parallel(n_jobs=n_jobs, verbose=1)

    parallel(
        delayed(_classify_pcap)(pcap, clf)
        for pcap in pcaps
        )


def _classify_pcap(pcap, clf):
    """
    res has the structure: domain -> [lbl (int/bool), list] ; where list contains src, dst, when tuples and lbl is 0/1
    :param pcap:
    :return:
    """
    log.info('Starting handling pcap: {!s}'.format(pcap))

    tlds = PublicSuffixes().get_valid_tlds()
    res = dict()

    for domain, src, dst, when in pcap_iterator(pcap):
        if '.' + domain.split('.')[-1] in tlds:
            if domain not in res:
                res[domain] = [-1, [(src, dst, when)]]
            else:
                res[domain][1].append((src, dst, when))

    domains = list(res.keys())

    prediction = clf.predict(domains)

    for i in range(len(domains)):
        res[domains[i]][0] = prediction[i]

    log.info('Filtering dict of len {!s}'.format(len(res)))
    for d in list(res.keys()):
        if res[d][0] != 1:
            del res[d]
    log.info('Finished filtering new len {!s}'.format(len(res)))

    log.info('Starting FP reduction...')
    to_remove = data.fp_reduction_alexa_top1m(list(res.keys()), count=1000000, return_removed=True)
    for d in to_remove:
        del res[d]
    to_remove = data.fp_reduction_local_domains(list(res.keys()), siemens=False, return_removed=True)
    for d in to_remove:
        del res[d]
    log.info('Finished FP reduction...')

    serialize_efficient(settings.PCAP_CLASSIFICATION_FOLDER + '{!s}.pkl'.format(ntpath.basename(pcap)), res)

    log.info('Finished pcap: {!s}'.format(pcap))


def serialize_efficient(where, what, keep_copy=True, chunked=True):
    """
    Pickles given py obj. (what) to given file (where)
    If file exists: keeps a copy if not turned of via keep_copy=False
    :param where:
    :param what:
    :param keep_copy:
    :return:
    """
    if os.path.isfile(where):
        if not keep_copy:
            return
        where += '_copy_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

    with open(where, 'wb') as f:
        joblib.dump(what, f)


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}
