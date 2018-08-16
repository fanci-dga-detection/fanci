import logging
import os
import pickle

from collections import defaultdict

import numpy
from sklearn import preprocessing
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.externals.joblib.parallel import Parallel, delayed
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneGroupOut, cross_val_score
from sklearn.svm import SVC

import settings

from learning import stats_metrics, classifiers
from learning.classifiers import SVMClassifier, RFClassifier, Clf
from learning.feature_extraction import extract_all_features
from learning.stats_metrics import Statistic
from settings import GRID_SEARCH_FOLDER, NOW_STR, COPY_SUFFIX, SCORE_SUFFIX
from data_processing import data
from data_processing.data import DataSet, GroupedDataSet

log = logging.getLogger('log')


def rf_grid_search(data_set: DataSet, n_est_values=16, n_jobs=8):
    num_of_estimators = numpy.random.random_integers(10, 1000, n_est_values)
    max_feat = range(2, 44)
    param_grid = dict(max_features=max_feat, n_estimators=num_of_estimators, criterion=['gini', 'entropy'])
    return grid_search(RandomForestClassifier(), param_grid, data_set, n_jobs=n_jobs)


def svm_grid_search(data_set: DataSet, n_values=32, search_start_exp=-16, search_end_exp=3, n_jobs=8):
    c_range = numpy.logspace(search_start_exp, search_end_exp, n_values, base=2)
    gamma_range = numpy.logspace(search_start_exp, search_end_exp, n_values, base=2)
    param_grid_lin = dict(C=c_range, kernel=['linear'])
    param_grid_rbf = dict(gamma=gamma_range, C=c_range, kernel=['rbf'])
    param_grid = [param_grid_lin, param_grid_rbf]
    return grid_search(SVC(), param_grid, data_set, n_jobs=n_jobs)


def predict_all_on_x(clf: Clf, data_set: GroupedDataSet, dmns=None, lbls=None, groups=None):
    """
    Performs prediction of clf using data_set.
    :param clf: ready-to-use trained classifier for one single DGA
    :param data_set: a grouped data set
    :param persist: save results to disk
    :param dga: DGA clf is trained on
    :return:
    """

    if not None:
        dmns, lbls, groups = data_set.expand()

    for k in data_set.family_map.keys():
        if clf.dga == data_set.family_map[k]:
            target_dga_group = k
            break

    for i in range(len(lbls)):
        if lbls[i] == 1 and groups[i] != target_dga_group:
            lbls[i] = 2

    _, pred_lbl = clf.predict(dmns, lbls)

    return clf.clf_type, clf.dga, lbls, pred_lbl


def grid_search(clf, param_grid, data_set: DataSet, persist=True, n_splits=5, n_jobs=8):

    log.info('Starting automated grid search. With {!s} jobs. Set {!s}'.format(n_jobs, data_set.id))
    domains, labels = data_set.expand()

    log.info('Set size is: {!s}. Parameter ranges are: {!s}'.format(len(data_set.full), param_grid))

    feature_matrix = extract_all_features(domains)
    if isinstance(clf, SVC):
        feature_matrix = preprocessing.StandardScaler().fit_transform(feature_matrix)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)

    grid = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=n_jobs, scoring=stats_metrics.multi_scorer_gridsearch, refit=False, verbose=2, return_train_score=False) # turn refit to True to be able to reuse the best estimator
    grid.fit(feature_matrix, labels)

    log.info('Grid search finished.')
    log.info('Full statistics: \n {!s}'.format(grid.cv_results_))
    log.info('Best parameters choice: {!s} with score: {!s}'.format(grid.best_params_, grid.best_score_))

    with open(settings.GRID_SEARCH_FOLDER_TMP + ''.join(stats_metrics.get_rand_id()), 'rb') as f:
        all_scores = pickle.load(f)

    if persist:
        grid_str = '__'

        if isinstance(clf, SVC):
            for g in param_grid:
                for k in g.keys():
                    grid_str += '{!s}{!s}_'.format(k, len(g[k]))
        else:
            for k in param_grid.keys():
                grid_str += '{!s}{!s}_'.format(k, len(param_grid[k]))


        grid_str += 'params{!s}__'.format(len(param_grid))
        grid_file = GRID_SEARCH_FOLDER + '{!s}.pkl'.format(data_set.id + grid_str + NOW_STR)
        if os.path.isfile(grid_file):
            grid_file += COPY_SUFFIX + ''.join(stats_metrics.get_rand_id())
        data_set.serialize(keep_copy=False)
        with open(grid_file, 'wb+') as f:
            pickle.dump(grid, f)
        with open(grid_file + SCORE_SUFFIX, 'wb+') as f:
            pickle.dump(all_scores, f)

    return grid, all_scores


def leave_one_group_out_deprecated(clf, data_set: GroupedDataSet, n_jobs=8):
    log.info('Starting leave on group out cv.')
    logo = LeaveOneGroupOut()
    domains, labels, groups = data_set.expand()
    log.info('Set dimensions: {!s} x {!s} x {!s}'.format(len(domains), len(labels), len(groups)))
    log.info('Starting feature extraction.')
    feature_matrix = extract_all_features(domains)
    if isinstance(clf, SVC):
        std_scale = preprocessing.StandardScaler()
        feature_matrix = std_scale.fit_transform(feature_matrix)

    log.info('Feature extraction finished.')

    scores = cross_val_score(clf, feature_matrix, labels, groups, cv=logo, scoring=stats_metrics.multi_scorer_gridsearch, n_jobs=n_jobs, verbose=2)
    return scores


def logo_cv(clf_type, data_sets: [GroupedDataSet], n_jobs=-1, parallel_verbose=1, persist=True):
    """
    Parallel leave on group out cross validation.
    :param clf:
    :param data_sets:
    :param n_jobs:
    :param parallel_verbose:
    :param persist:
    :return:
    """
    log.info('Starting leave on group out cv for {!s} sets'.format(len(data_sets)))

    parallel = Parallel(n_jobs=n_jobs, verbose=parallel_verbose)

    logo = LeaveOneGroupOut()
    stats_list = parallel(delayed(_fit_and_score)(clf, domains, labels, train_index, test_index, -1, data_set_id, -1)
                      for domains, labels, groups, data_set_id, clf in _grouped_data_sets_generator(data_sets, clf_type)
                      for train_index, test_index in logo.split(domains, labels, groups=groups))
    where = settings.EVAL_FOLDER + '/' + 'logo_cv_{!s}_{!s}sets_{!s}.pkl'.format(clf_type, len(data_sets),
                                                                                                settings.NOW_STR)
    return _serialize_cv_results(stats_list, persist, where)


def kfold_cv(clf_type, data_sets: [DataSet], fold_count=5, repetitions=5, n_jobs=-1, parallel_verbose=1, persist=True):
    """
    Do a kfold cross validation with a SVM classifier.
    :param data_sets: list of data sets
    :param fold_count: count of folds to be made and hence also runs
    :return: a Statistics object
    """
    log.info('Starting {!s}-fold cv. Set count: {!s}'.format(fold_count, len(data_sets)))
    parallel = Parallel(n_jobs=n_jobs, verbose=parallel_verbose)

    skf = StratifiedKFold(n_splits=fold_count, shuffle=True)
    stats_list = parallel(delayed(_fit_and_score)(clf, domains, labels, train_index, test_index, i, data_set_id, fold_count)
                          for domains, labels, data_set_id, clf in _data_sets_generator(data_sets, clf_type)
                          for i in range(repetitions)
                          for train_index, test_index in skf.split(domains, labels)
                          )
    where = settings.EVAL_FOLDER + '/' + '{!s}fold_cv_{!s}_{!s}rep_{!s}sets_{!s}.pkl'.format(fold_count, clf_type, repetitions, len(data_sets),
                                                                                                settings.NOW_STR)
    return _serialize_cv_results(stats_list, persist, where)


def trained_clfs_vs_sets(clfs: [Clf], data_sets: [DataSet], n_jobs=-1, parallel_verbose=1, persist=True):
    parallel = Parallel(n_jobs=n_jobs, verbose=parallel_verbose)
    stats_list = parallel(
        delayed(_score)(c, domains, labels, data_set_id)
        for c in clfs
        for domains, labels, data_set_id, _ in _data_sets_generator(data_sets, c.clf_type + '_' + c.dga)
        )

    where = settings.EVAL_FOLDER + '/' + 'rwth_vs_siemens_cv_{!s}clfs_{!s}sets_{!s}.pkl'.format(len(clfs), len(data_sets),
                                                                                                settings.NOW_STR)
    return _serialize_cv_results(stats_list, persist, where)


def _serialize_cv_results(stats_list, persist, where):
    res_stats_map = defaultdict(list)
    for s, data_set_id in stats_list:
        res_stats_map[data_set_id].append(s)
    if persist:
        data.serialize_keep_copy(where, res_stats_map, keep_copy=True)
    return res_stats_map


def _grouped_data_sets_generator(data_sets: [GroupedDataSet], clf_type):
    # XXX fix the magic bullshit SETTINGS array access fuck up
    for s in data_sets:
        domains, labels, groups = s.expand()

        if s.type == settings.SetTypes.mixed_dga_grouped_family.value:
            dga_type = 'mix'
        elif s.type == settings.SetTypes.single_dga_grouped_seed.value:
            dga_type = classifiers.DGA_TYPE_MAP[s.families[0].split('_')[0]]
        elif s.type == settings.SetTypes.single_dga_grouped_day.value:
            dga_type = classifiers.DGA_TYPE_MAP[s.id.split('_')[5]]

        if clf_type == 'svm':
            clf = SVMClassifier(dga_type=dga_type)
        elif clf_type == 'rf':
            clf = RFClassifier(dga_type=dga_type)
        else:
            log.error('Unknown classifier type given: {!s}'.format(clf_type))

        yield domains, labels, groups, s.id, clf


def _data_sets_generator(data_sets: [DataSet], clf_type):
    """
    Required for the subsequent generation of test and training data in the parallelized logo and cv.
    :param data_sets:
    :param clf_type:
    :return:
    """
    for s in data_sets:
        if 'mix' in clf_type:
            domains, labels, _ = s.expand()
            dga_type = 'mix'
        else:
            domains, labels = s.expand()
            dga_type = classifiers.DGA_TYPE_MAP[s.families[0]]
        if 'svm' in clf_type:
            clf = SVMClassifier(dga=dga_type)
        elif 'rf' in clf_type:
            clf = RFClassifier(dga=dga_type)
        else:
            log.error('Unknown classifier type given: {!s}'.format(clf_type))

        yield domains, labels, s.id, clf


def _score(clf, domains, labels, data_set_id):
    clf_type = clf.clf_type

    y_true, y_pred = clf.predict(domains, labels)

    stats = Statistic(set_id=data_set_id, id='rwth_vs_siemens_cv_{!s}_{!s}'.format(clf_type, data_set_id))

    stats.add_run(y_true, y_pred, domains)
    log.verbose('Truth vs. Prediction: \n' + str(list(y_true)) + '\n' + str(list(y_pred)))
    log.debug('\n' + classification_report(y_true, y_pred, target_names=['Benign', 'Malicious']))
    log.debug('\n' + str(confusion_matrix(y_true, y_pred)))
    log.debug('Missclassifications: {!s}'.format(stats.missclassified))

    return stats, data_set_id


def _fit_and_score(clf, domains, labels, train_index, test_index, repetition, data_set_id, fold_count):
    log.debug('Train index: {!s}\nTest index: {!s}'.format(train_index, test_index))

    clf_type = clf.clf_type

    clf.training(domains[train_index], labels[train_index])
    y_true, y_pred = clf.predict(domains[test_index], labels[test_index])
    if fold_count == -1:
        stats = Statistic(set_id=data_set_id,
                          id='logo_cv_{!s}_{!s}'.format(clf_type, data_set_id))
    else:
        stats = Statistic(set_id=data_set_id,
                          id='{!s}fold_cv_{!s}_rep{!s}_{!s}'.format(fold_count, clf_type, repetition, data_set_id))

    stats.add_run(y_true, y_pred, domains[test_index])
    log.verbose('Truth vs. Prediction: \n' + str(list(y_true)) + '\n' + str(list(y_pred)))
    log.debug('\n' + classification_report(y_true, y_pred, target_names=['Benign', 'Malicious']))
    log.debug('\n' + str(confusion_matrix(y_true, y_pred)))
    log.debug('MIssclassifications: {!s}'.format(stats.missclassified))

    return stats, data_set_id
