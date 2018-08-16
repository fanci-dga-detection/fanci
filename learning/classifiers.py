import sys
import logging
from os import walk

import pickle

from sklearn.externals.joblib.parallel import Parallel, delayed

import settings
from data_processing import data

sys.path.append('../')

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm, preprocessing

from learning.feature_extraction import extract_all_features, extract_all_features_single
from abc import ABC, abstractmethod

log = logging.getLogger('log')

# DGA families mapped to their respective type. Required for appropriate parameter selection.
DGA_TYPE_MAP = {'bamital': 'tdd-h',
                'banjori': 'tid-a',
                'bedep': 'tdn-a',
                'blackhole': 'tdd-a',
                'bobax': 'tid-a',
                'chinad': 'tid-a',
                'chir': 'tid-h',
                'conficker': 'tdd-a',
                'corebot': 'tdd-a',
                'cryptolocker': 'tdd-a',
                'dircrypt': 'tid-a',
                'dyre': 'tdd-h',
                'ekforward': 'tid-h',
                'emotet': 'tdd-a',
                'feodo': 'tid-a',
                'fobber': 'tid-a',
                'gameover': 'tdd-a',
                'gozi': 'tdd-w',
                'goznym': 'tdd-w',
                'gspy': 'tid-h',
                'hesperbot': 'tid-a',
                'locky': 'tdd-a',
                'madmax': 'tdd-a',
                'matsnu': 'tdd-w',
                'mirai': 'tdd-a',
                'murofet': 'tdd-a',
                'murofetweekly': 'tdd-a',
                'necurs': 'tdd-a',
                'nymaim': 'tdd-a',
                'oderoor': 'tid-a',
                'omexo': 'tid-h',
                'padcrypt': 'tid-a',
                'pandabanker': 'tid-h',
                'proslikefan': 'tdd-a',
                'pushdo': 'tdd-a',
                'pushdotid': 'tid-a',
                'pykspa': 'tdd-a',
                'pykspa2': 'tdd-a',
                'pykspa2s': 'tdd-a',
                'qadars': 'tid-a',
                'qakbot': 'tdd-a',
                'ramdo': 'tid-a',
                'ramnit': 'tid-a',
                'ranbyus': 'tdd-a',
                'redyms': 'tid-a',
                'rovnix': 'tid-a',
                'shifu': 'tid-a',
                'simda': 'tid-a',
                'sisron': 'tdd-a',
                'sphinx': 'tid-a',
                'suppobox': 'tdd-w',
                'sutra': 'tid-a',
                'symmi': 'tid-a',
                'szribi': 'tdd-a',
                'tempedreve': 'tid-a',
                'tinba': 'tid-a',
                'tofsee': 'tid-a',
                'torpig': 'tdd-a',
                'ud2': 'tid-h',
                'urlzone': 'tid-a',
                'vawtrak': 'tid-a',
                'vidro': 'tdd-a',
                'vidrotid': 'tid-a',
                'virut': 'tdd-a',
                'volatilecedar': 'tid-p',
                'xxhex': 'tdd-h',
                'mix': 'mix'
                }


class Clf(ABC):

    def __init__(self, dga_type, dga):
        self.clf = None
        self.clf_type = ''
        self.param_map = {}
        self.dga_type = dga_type
        self.dga = dga

    @abstractmethod
    def predict(self, test, labels):
        pass

    def serialize(self, to=settings.CLF_FOLDER, id=''):
        """
        Write classifier to disk.
        :param to: folder where to save
        :return:
        """
        to += '/{!s}_{!s}_{!s}.pkl'.format(self.dga, self.clf_type, id)
        data.serialize_keep_copy(to, self)


class SVMClassifier(Clf):
    """
    SVM classifier providing training and prediction methods
    Conficker
    {'C': 0.172941310318, 'kernel': 'linear'}
    Corebot
      {'C': 3.46691098185, 'kernel': 'linear'}
    Dyre
      {'C': 0.00521803204349, 'kernel': 'linear'}
    Matsnu
      {'C': 0.228935940421, 'kernel': 'linear'}
    Oderoor
      {'C': 4.27296676326, 'kernel': 'linear'}
    Mix
        {'C': 0.915998868622, 'gamma': 0.0198017555736, 'kernel': 'rbf'}

    TDD-A (Corebot)
	TDD-H (Dyre)
	TID-A (Kraken / Oderoor)
	TDD-W (Matsnu)
	TDD-A (Conficker)

    """

    def __init__(self, dga='mix', params=None):
        super(SVMClassifier, self).__init__(DGA_TYPE_MAP[dga], dga)
        self.param_map = {'mix': {'C': 0.915998868622, 'gamma': 0.0198017555736, 'kernel': 'rbf'},
                     'tdd-a': {'C': 0.172941310318, 'kernel': 'linear'},
                     'tdn-a': {'C': 0.172941310318, 'kernel': 'linear'},
                     'tdd-h': {'C': 0.00521803204349, 'kernel': 'linear'},
                     'tid-h': {'C': 0.00521803204349, 'kernel': 'linear'},
                     'tid-a': {'C': 4.27296676326, 'kernel': 'linear'},
                     'tid-p': {'C': 0.023393936925383451, 'gamma': 0.032651456065059363, 'kernel': 'rbf'},
                     'tdd-w': {'C': 0.228935940421, 'kernel': 'linear'},
                     'tid-w': {'C': 0.228935940421, 'kernel': 'linear'}
                     }
        if params:
            self.clf = svm.SVC(**params)
        else:
            self.clf = svm.SVC(**self.param_map[DGA_TYPE_MAP[dga]])
        self.std_scale = preprocessing.StandardScaler()
        self.clf_type = 'svm'

    def training(self, train, labels):
        """
        Training on given data.
        :param train: array-like containing domain name strings
        :param labels: array-like containing labels
        :return: void
        """

        log.info('Starting training of svm with training set of cardinality: {!s}'.format(len(train)))

        feature_matrix = extract_all_features(train)
        feature_matrix = self.std_scale.fit_transform(feature_matrix)

        self.clf.fit(feature_matrix, labels)

    def predict(self, test, labels=None):
        """
        Predict test data
        :param test: array of samples to predict
        :return: array of true labels, array of predicted labels
        """

        feature_matrix = extract_all_features(test)
        log.info('Starting prediction of {!s} samples'.format(len(test)))

        feature_matrix = self.std_scale.transform(feature_matrix)
        prediction = self.clf.predict(feature_matrix)
        if labels is not None:
            return labels, prediction
        else:
            return prediction

    def predict_single_preprocessed(self, s):
        s = self.std_scale.transform(s)
        return self.clf.predict(s)[0]

    def predict_bulk_preprocessed(self, feature_matrix):
        feature_matrix = self.std_scale.transform(feature_matrix)
        return self.clf.predict(feature_matrix)


class RFClassifier(Clf):
    """
    Encapsulates a sklearn RFEnsableClassifier.
    Conficker
      {'criterion': 'entropy', 'max_features': 16, 'n_estimators': 23}
    Corebot
      {'criterion': 'gini', 'max_features': 8, 'n_estimators': 681}
    Dyre
      {'criterion': 'gini', 'max_features': 2, 'n_estimators': 388}
    Matsnu
      {'criterion': 'gini', 'max_features': 5, 'n_estimators': 57}
    Oderoor
      {'criterion': 'gini', 'max_features': 7, 'n_estimators': 36}
    Mix
      {'criterion': 'gini', 'max_features': 18, 'n_estimators': 785}
    TDD-A (Corebot)
	TDD-H (Dyre)
	TID-A (Kraken / Oderoor)
	TDD-W (Matsnu)
	TDD-A (Conficker)
    """

    def __init__(self, dga='mix', params=None):
        super(RFClassifier, self).__init__(DGA_TYPE_MAP[dga], dga)
        self.param_map = {'mix': {'criterion': 'gini', 'max_features': 18, 'n_estimators': 785},
                     'tdd-a': {'criterion': 'entropy', 'max_features': 16, 'n_estimators': 23},
                     'tdn-a': {'criterion': 'entropy', 'max_features': 16, 'n_estimators': 23},
                     'tdd-h': {'criterion': 'gini', 'max_features': 2, 'n_estimators': 388},
                     'tid-h': {'criterion': 'gini', 'max_features': 2, 'n_estimators': 388},
                     'tid-a': {'criterion': 'gini', 'max_features': 7, 'n_estimators': 36},
                     'tid-p': {'criterion': 'gini', 'max_features': 2, 'n_estimators': 513},
                     'tdd-w': {'criterion': 'gini', 'max_features': 5, 'n_estimators': 57},
                     'tid-w': {'criterion': 'gini', 'max_features': 5, 'n_estimators': 57}
                     }
        if params:
            self.clf = RandomForestClassifier(**params)
        else:
            self.clf = RandomForestClassifier(**self.param_map[DGA_TYPE_MAP[dga]])
        self.clf_type = 'rf'

    def training(self, train, labels):
        """
        Training on given data.
        :param train: array-like containing domain name strings
        :param labels: array-like containing labels
        :return: void
        """
        log.info('Starting training of RF classifier with training set of cardinality: {!s}'.format(len(train)))

        feature_matrix = extract_all_features(train)
        self.clf.fit(feature_matrix, labels)

    def predict(self, test, labels=None):
        """
        Predict test data
        :param test: array of samples to predict
        :return: array of true labels, array of predicted labels
        """

        log.info('Starting prediction of {!s} samples'.format(len(test)))

        feature_matrix = extract_all_features(test)
        prediction = self.clf.predict(feature_matrix)

        if labels is not None:
            return labels, prediction
        else:
            return prediction

    def predict_bulk_preprocessed(self, feature_matrix):
        return self.clf.predict(feature_matrix)


def predict_single_preprocessed(self, s):
        return self.clf.predict(s)[0]


class ClassificationEnsemble:

    def __init__(self, clf_list: [Clf] = None, nomix=False, only_type=None, only_mix=False, clfs_from_path=None):
        self.clfs = []
        if clf_list:
            self.clfs = clf_list
        else:
            log.info('Loading all available classifiers.')
            self.__load_clfs(clfs_from_path=clfs_from_path)
        if only_mix:
            self.clfs = [c for c in self.clfs if 'mix' in c.dga_type]
        if only_type:
            self.clfs = [c for c in self.clfs if only_type == c.clf_type]

        if nomix:
            self.clfs = [c for c in self.clfs if 'mix' not in c.dga_type]

    def __load_clfs(self, clfs_from_path=None):
        """
        Loads all serialized classifiers from the disk using the default folder
        :return:
        """

        if not clfs_from_path:
            clfs_from_path = settings.CLF_FOLDER

        clfs_files = []
        for (path, dirs, files) in walk(clfs_from_path):
            clfs_files.extend(files)
            break

        clfs_files = [clfs_from_path + '/' + f for f in clfs_files]

        for c in clfs_files:
            with open(c, 'rb') as f:
                self.clfs.append(pickle.load(f))

    def predict_threshold(self, domains, lbls=None, threshold=1, n_jobs=-1):
        """
        Predict on all clfs combined by mathematical or.
        :param domains: iterable containing domains as str
        :param n_jobs:
        :return:
        """
        domains = list(domains)
        merged = self.predict_all_plain(domains, n_jobs)

        votemap = self.results_to_vote_map(domains, merged)

        self._votemap_to_label(threshold, votemap)

        return lbls, votemap

    def _votemap_to_label(self, threshold, votemap):
        for i in range(len(votemap)):
            if votemap[i] >= threshold:
                votemap[i] = 1
            else:
                votemap[i] = 0

    def _votemap_to_label_exact_or_threshold(self, threshold, votemap, exact=1):
        for i in range(len(votemap)):
            if votemap[i] >= threshold:
                votemap[i] = 1
            elif votemap[i] == exact:
                votemap[i] = 1
            else:
                votemap[i] = 0

    def predict_duo(self, domains, lbls=None, n_jobs=-1):
        """
        Considers as malicious if and only if svm and rf clf vote for malicious
        :return:
        """
        domains = list(domains)
        merged = self.predict_all_plain(domains, n_jobs)
        votemap = self.results_to_duo_votemap(domains, merged)

        self._votemap_to_label(0, votemap)

        return lbls, votemap

    def predict_duo_threshold(self, domains, lbls=None, threshold=1, n_jobs=-1):
        """
        Considers as malicious if and only if svm and rf clf vote for malicious
        :return:
        """
        domains = list(domains)
        merged = self.predict_all_plain(domains, n_jobs)
        votemap = self.results_to_duo_votemap(domains, merged)

        self._votemap_to_label(threshold, votemap)

        return lbls, votemap

    def results_to_duo_votemap(self, domains, merged):
        votemap = [0] * len(domains)
        for i in range(len(domains)):
            for dga, clf_type in merged:
                if merged[dga, 'rf'][i] == 1 and merged[dga, 'svm'][i] == 1:
                    votemap[i] += 1
        return votemap

    def results_to_vote_map(self, domains, merged):
        votemap = [0]*len(domains)
        for i in range(len(domains)):
            for dga, clf_type in merged:
                votemap[i] += merged[dga, clf_type][i]

        return votemap

    def predict_exactly_one_or_threshold(self, domains, lbls=None, threshold=3, exact=1, n_jobs=-1):
        """
        Predict on all clfs combined by mathematical or.
        :param domains: iterable containing domains as str
        :param n_jobs:
        :return:
        """
        domains = list(domains)
        merged = self.predict_all_plain(domains, n_jobs)
        votemap = self.results_to_vote_map(domains, merged)

        self._votemap_to_label_exact_or_threshold(threshold, votemap, exact=exact)

        return lbls, votemap

    def predict_all_plain(self, domains, n_jobs=-1):
        """
        Predicts the label of d using all classifiers present
        :param domains: iterable containing domains as str
        :return: dictionry containing results (dga, svm/rf):label
        """
        feature_matrix = extract_all_features(domains)

        parallel = Parallel(n_jobs=n_jobs, verbose=1)

        res = parallel(
            delayed(_predict)(c, feature_matrix)
            for c in self.clfs
        )
        # TODO
        merged = {}
        for d in res:
            keys = list(d.keys())
            if keys[0] in res:
                merged[keys[0]] += d[keys[0]]
            else:
                merged.update(d)
        return merged


def _predict(c, feature_matrix):
    log.info('Predicting {!s} samples with {!s}'.format(len(feature_matrix), c.clf_type))
    return {(c.dga, c.clf_type): c.predict_bulk_preprocessed(feature_matrix)}
