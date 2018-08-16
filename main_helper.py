import random
import sys
from timeit import Timer

from os.path import expanduser
from sklearn.externals.joblib.parallel import Parallel, delayed

import settings
from data_processing import data
from learning.stats_metrics import Statistic
from main import classification
from learning import eval_train_test, feature_extraction, classifiers
from learning.classifiers import SVMClassifier, RFClassifier

sys.path.append('./')

from data_processing.data import Workspace


def predict_all_pcaps():
    classification.classify_pcaps('/work/ss930620/dnscap/')


def test_all_chains():
    nomix =[True, False]
    clf_type = ['svm', 'rf', None]
    exact = [1, 2, 3, 4]
    threshold = [1, 2, 3, 4, 5, 6]

    results = dict()

    for c in clf_type:
        for m in nomix:
            for t in threshold:
                results[('threshold', c, m, t)] = predict_threshold(clf_type=c, nomix=m, threshold=t)

    for m in nomix:
        for t in threshold:
            results[('duo_threshold', c, m, t)] = predict_duo_threshold(nomix=m, threshold=t)

    for m in nomix:
        results[('duo', c, m)] = predict_duo(nomix=m)

    for c in clf_type:
        for m in nomix:
            for t in threshold:
                for e in exact:
                    results[('exact_threshold', c, m, t, e)] = predict_exact_or_threshold(nomix=m, clf_type=c, threshold=t, exact=e)

    threshold = [1, 2]
    for t in threshold:
        results[('onlymix_threshold', t)] = predict_threshold(nomix=False, only_mix=True, threshold=t)

    data.serialize_keep_copy(settings.ANALYSIS_FOLDER + '/chaining.pkl', results)
    return results


def test_rf_mix_chains():
    results = dict()
    threshold = [1, 2]
    clfs_all = classifiers.ClassificationEnsemble(nomix=False, only_type='rf', only_mix=True).clfs
    for i in range(3, 19):
        for t in threshold:
            clfs_sel = classifiers.ClassificationEnsemble(nomix=False, only_type='rf', only_mix=True,clf_list=random.sample(clfs_all, i))
            results[('mix_rf', t, i)] = predict_threshold(nomix=False, only_mix=True, threshold=t, clfs=clfs_sel)

    data.serialize_keep_copy(settings.ANALYSIS_FOLDER + '/chaining.pkl', results)
    return results


def predict_exact_or_threshold(clf_type=None, nomix=True, threshold=1, exact=1):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.mixed_dga_grouped_family.value)

    clfs = classifiers.ClassificationEnsemble(nomix=nomix, only_type=clf_type, only_mix=False)

    ds = w.data_sets_loaded['mixed_dga_grouped_family_50000_59_2.pkl']
    dmns, lbls, groups = ds.expand()

    lbls, predicted = clfs.predict_exactly_one_or_threshold(dmns, lbls, threshold=threshold, exact=exact)

    stats = Statistic(set_id='mixed_dga_grouped_family_50000_59_2.pkl',
                      id='threshold_test')

    stats.add_run(lbls, predicted, dmns)

    return stats


def predict_threshold(clf_type=None, nomix=True, threshold=1, only_mix=False, clfs=None):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.mixed_dga_grouped_family.value)

    if not clfs:
        clfs = classifiers.ClassificationEnsemble(nomix=nomix, only_type=clf_type, only_mix=only_mix)

    ds = w.data_sets_loaded['mixed_dga_grouped_family_50000_59_2.pkl']
    dmns, lbls, groups = ds.expand()

    lbls, predicted = clfs.predict_threshold(dmns, lbls, threshold=threshold)

    stats = Statistic(set_id='mixed_dga_grouped_family_50000_59_2.pkl',
                      id='threshold_test')

    stats.add_run(lbls, predicted, dmns)

    return stats


def predict_duo(nomix=True, only_mix=False):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.mixed_dga_grouped_family.value)

    clfs = classifiers.ClassificationEnsemble(nomix=nomix, only_mix=only_mix)

    ds = w.data_sets_loaded['mixed_dga_grouped_family_50000_59_2.pkl']
    dmns, lbls, groups = ds.expand()

    lbls, predicted = clfs.predict_duo(dmns, lbls)

    stats = Statistic(set_id='mixed_dga_grouped_family_50000_59_2.pkl',
                      id='threshold_test')

    stats.add_run(lbls, predicted, dmns)

    return stats


def predict_duo_threshold(nomix=True, threshold=1, only_mix=False):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.mixed_dga_grouped_family.value)

    clfs = classifiers.ClassificationEnsemble(nomix=nomix, only_mix=only_mix)

    ds = w.data_sets_loaded['mixed_dga_grouped_family_50000_59_2.pkl']
    dmns, lbls, groups = ds.expand()

    lbls, predicted = clfs.predict_duo_threshold(dmns, lbls, threshold=threshold)

    stats = Statistic(set_id='mixed_dga_grouped_family_50000_59_2.pkl',
                      id='threshold_test')

    stats.add_run(lbls, predicted, dmns)

    return stats


def predict_all_mixed_sets_on_x(n_jobs=8):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.mixed_dga_grouped_family.value)

    clfs = classifiers.ClassificationEnsemble().clfs

    clfs = [c for c in clfs if 'mix' not in c.dga_type]

    parallel = Parallel(n_jobs=n_jobs, verbose=1)

    ds = w.data_sets_loaded['mixed_dga_grouped_family_50000_59_2.pkl']
    dmns, lbls, groups = ds.expand()

    # res is list of tuples: clf.clf_type, clf.dga_type, lbls, pred_lbl
    res = parallel(
        delayed(eval_train_test.predict_all_on_x)(clf, ds, dmns, lbls, groups)
        for clf in clfs
    )

    data.serialize_keep_copy(settings.ANALYSIS_FOLDER + '/x_vs_all_results.pkl', (res, dmns))


def time_feature_extraction(rep=1):
    w = Workspace(days=1, empty=True)
    set_name = 'mixed_dga_grouped_family_50000_59_0.pkl'
    w.load(set_name, settings.SetTypes.mixed_dga_grouped_family.value)
    domains = [ld.domain for ld in w.data_sets_loaded['mixed_dga_grouped_family_50000_59_0.pkl'].full]

    t = Timer(lambda: feature_extraction.extract_all_features(domains))
    print(t.timeit(number=rep))


def time_training(rep=1):
    w = Workspace(days=1, empty=True)
    set_name = 'mixed_dga_grouped_family_50000_59_0.pkl'
    w.load(set_name, settings.SetTypes.mixed_dga_grouped_family.value)
    dom, lab, _ = w.data_sets_loaded['mixed_dga_grouped_family_50000_59_0.pkl'].expand()

    svm = SVMClassifier(dga='mix')
    t = Timer(lambda:  svm.training(dom, lab))
    print('SVM Training: ' + str(t.timeit(number=rep)))

    rf = RFClassifier(dga='mix')
    t = Timer(lambda:  rf.training(dom, lab))
    print('RF Training: ' + str(t.timeit(number=rep)))

    set_name = 'mixed_dga_grouped_family_50000_59_1.pkl'
    w.load(set_name, settings.SetTypes.mixed_dga_grouped_family.value)
    dom, lab, _ = w.data_sets_loaded['mixed_dga_grouped_family_50000_59_1.pkl'].expand()

    t = Timer(lambda:  svm.predict(dom, lab))
    print('SVM Classify: ' + str(t.timeit(number=rep)))

    t = Timer(lambda: rf.predict(dom, lab))
    print('RF Classify: ' + str(t.timeit(number=rep)))


def start_single_dga_kfold(repetitions=5, n_jobs=-1, clf_type=None):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.single_dga.value)

    if not clf_type:
        eval_train_test.kfold_cv('svm', w.data_sets_loaded.values(), repetitions=repetitions, n_jobs=n_jobs)
        eval_train_test.kfold_cv('rf', w.data_sets_loaded.values(), repetitions=repetitions, n_jobs=n_jobs)
    elif clf_type == 'rf':
        eval_train_test.kfold_cv('rf', w.data_sets_loaded.values(), repetitions=repetitions, n_jobs=n_jobs)
    elif clf_type == 'svm':
        eval_train_test.kfold_cv('svm', w.data_sets_loaded.values(), repetitions=repetitions, n_jobs=n_jobs)


def start_mix_dga_kfold(repetitions=5, n_jobs=-1, clf_type=None):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.mixed_dga_grouped_family.value)

    if not clf_type:
        eval_train_test.kfold_cv('svm_mix', w.data_sets_loaded.values(), repetitions=repetitions, n_jobs=n_jobs)
        eval_train_test.kfold_cv('rf_mix', w.data_sets_loaded.values(), repetitions=repetitions, n_jobs=n_jobs)
    elif clf_type == 'rf':
        eval_train_test.kfold_cv('rf_mix', w.data_sets_loaded.values(), repetitions=repetitions, n_jobs=n_jobs)
    elif clf_type == 'svm':
        eval_train_test.kfold_cv('svm_mix', w.data_sets_loaded.values(), repetitions=repetitions, n_jobs=n_jobs)


def start_mixed_dga_logo(n_jobs=-1, clf_type=None):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.mixed_dga_grouped_family.value)

    if not clf_type:
        eval_train_test.logo_cv('svm', w.data_sets_loaded.values(), n_jobs=n_jobs)
        eval_train_test.logo_cv('rf', w.data_sets_loaded.values(), n_jobs=n_jobs)
    elif clf_type == 'rf':
        eval_train_test.logo_cv('rf', w.data_sets_loaded.values(), n_jobs=n_jobs)
    elif clf_type == 'svm':
        eval_train_test.logo_cv('svm', w.data_sets_loaded.values(), n_jobs=n_jobs)


def start_single_grouped_seed_logo(n_jobs=-1, clf_type=None):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.single_dga_grouped_seed.value)

    if not clf_type:
        eval_train_test.logo_cv('svm', w.data_sets_loaded.values(), n_jobs=n_jobs)
        eval_train_test.logo_cv('rf', w.data_sets_loaded.values(), n_jobs=n_jobs)
    elif clf_type == 'rf':
        eval_train_test.logo_cv('rf', w.data_sets_loaded.values(), n_jobs=n_jobs)
    elif clf_type == 'svm':
        eval_train_test.logo_cv('svm', w.data_sets_loaded.values(), n_jobs=n_jobs)


def start_single_grouped_day_logo(n_jobs=-1, clf_type=None):
    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.single_dga_grouped_day.value)

    if not clf_type:
        eval_train_test.logo_cv('svm', w.data_sets_loaded.values(), n_jobs=n_jobs)
        eval_train_test.logo_cv('rf', w.data_sets_loaded.values(), n_jobs=n_jobs)
    elif clf_type == 'rf':
        eval_train_test.logo_cv('rf', w.data_sets_loaded.values(), n_jobs=n_jobs)
    elif clf_type == 'svm':
        eval_train_test.logo_cv('svm', w.data_sets_loaded.values(), n_jobs=n_jobs)


def start_x_trained_y_test(n_jobs=-1, clf_type=None, x='siemens'):

    if x == 'rwth':
        load_clfs_from = '/work/ss930620/dga_detection_workspace/clfs/'
    elif x == 'siemens':
        load_clfs_from = '/work/ss930620/dga_detection_workspace_siemens/clfs/'

    w = Workspace(days=1, empty=True)
    w.load_all(settings.SetTypes.mixed_dga_grouped_family.value)
    test_data_sets = w.data_sets_loaded.values()

    if not clf_type:
        eval_train_test.trained_clfs_vs_sets(classifiers.ClassificationEnsemble(only_mix=True, clfs_from_path=load_clfs_from).clfs, test_data_sets, n_jobs=n_jobs)
    elif clf_type == 'rf':
        eval_train_test.trained_clfs_vs_sets(
            classifiers.ClassificationEnsemble(only_mix=True, only_type='rf', clfs_from_path=load_clfs_from).clfs, test_data_sets,
            n_jobs=n_jobs)
    elif clf_type == 'svm':
        eval_train_test.trained_clfs_vs_sets(
            classifiers.ClassificationEnsemble(only_mix=True, only_type='svm', clfs_from_path=load_clfs_from).clfs,
            test_data_sets,
            n_jobs=n_jobs)
