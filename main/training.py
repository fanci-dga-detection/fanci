import random

from sklearn.externals.joblib.parallel import Parallel, delayed

import settings
from data_processing.data import Workspace, GroupedDataSet
from learning import classifiers

backlog = set()


def train_all_available_dga(clf_type=None):
    """
    Trains all available single DGAs.
    :param clf_type: 'svm', 'rf', None (None means for both)
    :return:
    """
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.single_dga.value)

    if not clf_type:
        for ct in ['svm', 'rf']:
            for s in w.data_sets_loaded.values():
                _training(ct, s)
    elif clf_type == 'rf':
        for s in w.data_sets_loaded.values():
            _training(clf_type, s)
    elif clf_type == 'svm':
        for s in w.data_sets_loaded.values():
            _training(clf_type, s)


def train_mixed_dga(clf_types=['svm', 'rf'], n_jobs=-1):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.mixed_dga_grouped_family.value)

    parallel = Parallel(n_jobs=n_jobs, verbose=1)
    parallel(
        delayed(_training)(clf_type, s)
        for s in w.data_sets_loaded.values()
        for clf_type in clf_types
    )

def _training(clf_type, set):
    """
    Train a classifier on given set of given type and serializes it to default clfs folder.
    :param clf_type: either 'svm' or 'rf'
    :param set: loaded data set
    :return: void
    """
    dga = set.id.split('_')[-2]
    if dga not in classifiers.DGA_TYPE_MAP.keys():
        dga = 'mix'

    if clf_type == 'svm':
        clf = classifiers.SVMClassifier(dga)
    elif clf_type == 'rf':
        clf = classifiers.RFClassifier(dga)

    if isinstance(set, GroupedDataSet):
        dom, lab, _ = set.expand()
    else:
        dom, lab = set.expand()
    clf.training(dom, lab)
    clf.serialize(id=set.id)
