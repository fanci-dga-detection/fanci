import json
import logging
import multiprocessing
import random
import string

import os
from collections import defaultdict

import copy
import numpy
import pickle

import pandas
from pandas.core.frame import DataFrame
from sklearn.externals import joblib

import settings


from os import walk

from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics.classification import confusion_matrix, classification_report
from sklearn.svm.classes import SVC

from data_processing import data, dgarchive_api, pcap_domain_utils
from main import classification
from settings import GRID_SEARCH_FOLDER
from learning import classifiers

log = logging.getLogger('log')
__rand_id = random.choices(string.ascii_uppercase + string.digits, k=5)
__lock = multiprocessing.RLock()


def get_rand_id():
    return __rand_id


def reset_rand_id():
    global __rand_id
    __rand_id = random.choices(string.ascii_uppercase + string.digits, k=5)


class ClassificationResults:
    # res has the structure: domain -> [lbl (int/bool), list] ; where list contains src, dst, when tuples and lbl is 0/1
    def __init__(self, path=settings.PCAP_CLASSIFICATION_FOLDER):
        self.path = path
        self.file_paths = []
        self.results = []
        for (_, _, files) in walk(path):
            self.file_paths.extend(files)
            break

        self.file_paths = [settings.PCAP_CLASSIFICATION_FOLDER + '/' + f for f in self.file_paths]

        for file in self.file_paths:
            with open(file, 'rb') as f:
                self.results.append(joblib.load(f))

        self.merged_results = dict()

        for res in self.results:
            for d in res.keys():
                if d not in self.merged_results:
                    self.merged_results[d] = res[d]
                else:
                    self.merged_results[d][1].extend(res[d][1])

        self.all_domains = set(self.merged_results.keys())

    def summary(self):
        log.info('Overall positives: {!s}'.format(len(self.all_domains)))

        pos_t2 = len(self.threshold_domain(2))
        pos_t3 = len(self.threshold_domain(3))
        pos_t5 = len(self.threshold_domain(5))
        pos_t10 = len(self.threshold_domain(10))
        pos_t20 = len(self.threshold_domain(20))

        log.info('Threshold counts \n 2: {!s}\n 3: {!s} \n 5: {!s} \n 10: {!s}\n 20: {!s}\n'.format(pos_t2, pos_t3, pos_t5, pos_t10, pos_t20))

    def xld_histogram(self, x=-1):
        r = pcap_domain_utils.sort_domain_counter_dict(pcap_domain_utils.histogram_xld(self.all_domains), x=x)
        return r

    def search_for_sequence(self, sequence, suffix=False, prefix=False):
        results = set()
        for d in self.all_domains:
            if prefix:
                if d.startswith(sequence):
                    results.add(d)
            elif suffix:
                if d.endswith(sequence):
                    results.add(d)
            else:
                if sequence in d:
                    results.add(d)

        return results

    def hitlist_to_ipaddr(self, file, persist=False):
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        lines = {d.strip() for d in lines if d.strip()}

        results = dict()
        for d in lines:
            results[d] = self.merged_results[d][1]

        if persist:
            with open(settings.ANALYSIS_FOLDER + '/positives_ips.json', 'w') as f:
                f.write(json.dumps(results, sort_keys=True, indent=4, ))

        return results

    def ip_intersection(self, d1, d2):
        res_d1 = self.merged_results[d1][1]
        res_d2 = self.merged_results[d2][1]

        same_ip = set()

        for t1 in res_d1:
            for t2 in res_d2:
                if t1[1] == t2[1]:
                    same_ip.add(t1)

        return same_ip

    def ip_count_groups(self, file, persist=False):
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        lines = [d.strip() for d in lines if d.strip()]

        ud_groups = self.group(lines)

        results = defaultdict(set)
        for ud in ud_groups.keys():
            for d in ud_groups[ud]:
                for t in self.merged_results[d][1]:
                    results[ud].add(t[1])

        for k in results.keys():
            results[k] = list(results[k])

        if persist:
            with open(settings.ANALYSIS_FOLDER + '/positives_ip_counts.json', 'w') as f:
                f.write(json.dumps(results, sort_keys=True, indent=4, ))

        return results

    def group(self, lines):
        ud_groups = defaultdict(list)
        for l in lines:
            if l.startswith('UD'):
                cur_ud = l
            else:
                ud_groups[cur_ud].append(l)
        return ud_groups

    def match_dga_archive(self, file, persist=False):
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        res = dict()

        ud_groups = self.group(lines)
        for k in ud_groups.keys():
            g,h = dgarchive_api.analyze_qry_list(dgarchive_api.check_for_dga_domains(ud_groups[k], persist=False))
            res[k] = g,h
        if persist:
            with open(settings.ANALYSIS_FOLDER + '/positives_dgaarchive.json', 'w') as f:
                f.write(json.dumps(res, sort_keys=True, indent=4, ))
        return res

    def filter_again_and_write(self, t=0, grouptld=True):
        to_filter = ['.fev.de', '.cisco', '.onion']

        if t == 0:
            domains = self.all_domains
        else:
            domains = set(self.threshold_domain(threshold=t))

        to_remove = set()
        for d in domains:
            for s in to_filter:
                if d.endswith(s):
                    to_remove.add(d)

        domains.difference_update(to_remove)

        if grouptld:
            res = self.group_by_suffix(domains)
            with open(settings.ANALYSIS_FOLDER + '/positives.json', 'w') as f:
                f.write(json.dumps(res, sort_keys=True, indent=4, ))
            return res

        else:
            with open(settings.ANALYSIS_FOLDER + '/positives.txt', 'a+') as f:
                for d in domains:
                    f.write(d + ',\n')
            return domains

    def get_all_suffix(self, suf=''):
        domains = []
        for d in self.all_domains:
            if d.endswith(suf):
                domains.append(d)

        return domains

    def group_by_suffix(self, domains):

        groups = defaultdict(list)

        for d in domains:
            suf = d.split('.')[-1]
            groups[suf].append(d)

        for d in groups.keys():
            groups[d].sort()

        return groups

    def threshold_domain(self, threshold=2):
        results = []
        for d in self.all_domains:

            unique_ips = set()
            for t in self.merged_results[d][1]:
                unique_ips.add(t[1])

            if len(unique_ips) >= threshold:
                results.append(d)

        return results

    def threshold_ip(self, threshold=3):
        ip_counter = defaultdict(int)
        for d in self.all_domains:
            for t in self.merged_results[d][1]:
                ip_counter[t[1]] += 1

        for ip in list(ip_counter.keys()):
            if ip_counter[ip] < threshold:
                del ip_counter[ip]

        res = dict()
        for ip in ip_counter:
            for d in self.all_domains:
                for t in self.merged_results[d][1]:
                    if t[1] == ip:
                        if ip not in res:
                            res[ip] = ip_counter, [d]
                        else:
                            res[ip][1].append(d)

        return res

    def verfiy_dga_archive(self):

        dga_data = dgarchive_api.check_for_dga_domains(self.all_domains, persist=False)
        guess, hits = dgarchive_api.analyze_qry_list(dga_data)

        return guess, hits


class GridSearchResult(object):
    """
    Represents all grid search results.
    """
    def __init__(self, path):
        self.gridcv = open_pickle_file(path)
        self.set = os.path.basename(path.split('__')[0])
        self.scores = open_pickle_file(path + settings.SCORE_SUFFIX)
        self.mod_gridcv = self.build_full_dict()
        self.pandas_frame = pandas.DataFrame(self.mod_gridcv)
        self.pandas_frame = self.pandas_frame.rename(columns={'mean_test_score': 'mean_roc_score', 'std_test_score': 'stddev_roc_score'})
        if 'max_features' in path:
            self.clf = 'rf'
            self.pandas_frame.name = 'gridsearch_rf_{!s}'.format(self.set)
            self.dup_params = ['param_max_features', 'param_n_estimators', 'param_criterion']
        else:
            self.clf = 'svm'
            self.pandas_frame.name = 'gridsearch_svm_{!s}'.format(self.set)
            self.dup_params = ['param_kernel', 'param_gamma', 'param_C']

    def build_full_dict(self):
        """
        Matches the sklearn grid results file to the output of the multi_scorer by searching for the respective parameter sets.
        :return: modified and updated cv results dict. (works on a copy, does not touch the original dict)
        """
        # XXX add the scores for the seperate splits additionally to the data frame to make it complete
        cv = copy.deepcopy(self.gridcv.cv_results_)
        cv['mean_accuracy_score'] = []
        cv['mean_f1_score'] = []
        cv['mean_precision_score'] = []
        cv['mean_recall_score'] = []
        # cv['mean_roc_score'] = []
        for p in cv['params']:
            k = frozenset(p.items())
            s_list = self.scores[k]
            folds = len(s_list)
            res = defaultdict(int)
            for s in s_list:
                for k in s.keys():
                    res[k] += s[k]
            for k in res.keys():
                res[k] = res[k] / folds
            cv['mean_accuracy_score'].append(res['accuracy'])
            cv['mean_f1_score'].append(res['f1'])
            cv['mean_precision_score'].append(res['precision'])
            cv['mean_recall_score'].append(res['recall'])

        # cv['params'] = frozenset(cv['params']) # make it hashable to be able to index along parameters
            # cv['mean_roc_score'].append(res['roc']), roc score already included as "main score" criteria
        return cv

    def concat(self, search_res):
        """
        Concatenates two gridsearchresults to one pandas frame.
        :param search_res:
        :return:
        """
        # XXX also concat gridcv file.

        self.scores = {**self.scores, **search_res.scores}
        self.pandas_frame = self.pandas_frame.append(search_res.pandas_frame, ignore_index=True)
        self.pandas_frame.drop_duplicates(subset=self.dup_params, inplace=True)

    def print_top_n(self, n):
        """
        prints top n values according to the given score, 2nd criteria automatically considered is the std_deviation in the test scores of the single splits
        :param n:
        :return:
        """
        columns = ['params', 'mean_roc_score', 'mean_accuracy_score', 'mean_precision_score', 'mean_recall_score', 'mean_f1_score', 'stddev_roc_score', 'mean_fit_time']
        scoring = ['mean_roc_score', 'mean_accuracy_score', 'mean_precision_score', 'mean_recall_score', 'mean_f1_score']
        log.info('Set: {!s}, Classifier: {!s}'.format(self.set, self.clf))
        df_res = self.pandas_frame.nlargest(n, scoring)[columns]
        log.info('\n' + str(df_res))

        return df_res


class GridSearchEvaluation:
    """
    Allows to perform anaylsis and statistics on grid search results.
    """
    def __init__(self, list_of_files=None, merge_similars=True):
        """
        View on all present grid search result files in the workspace.
        Loads all from disk and is doing some preprocessing for analysis.
        :param list_of_files: defaults to all grid_files in the current workspace
        """
        self.file_paths = []
        self.grid_results = []
        log.info('Collecting grid search files.')
        for (_, _, files) in walk(GRID_SEARCH_FOLDER):
            self.file_paths.extend(files)
            break

        self.file_paths = [GRID_SEARCH_FOLDER + '/' + f for f in self.file_paths if not f.endswith(settings.SCORE_SUFFIX)]
        self.file_paths.sort()
        for f in self.file_paths:
            try:
                self.grid_results.append(GridSearchResult(f))
            except AttributeError:
                log.error('Grid file is corrupt or in the wrong form, it is skipped: {!s}'.format(f))

        log.info('{!s} grid search results successfully processed.'.format(len(self.grid_results)))
        if merge_similars:
            log.info('Merging all similar files.')
            self.__merge_all_similars()
            log.info('Merge complete. Merged results left: {!s}.'.format(len(self.grid_results)))

    def __merge_all_similars(self):
        similars = defaultdict(list)
        for g in self.grid_results:
            similars[(g.set, g.clf)].append(g)

        for k in similars.keys():
            s_list = similars[k]
            s_list = list(set(s_list))
            if len(s_list) > 1:
                base_gs = s_list[0]
                for i in range(1, len(s_list)):
                    base_gs.concat(s_list[i])
                    self.grid_results.remove(s_list[i])

    def available(self):
        for g in self.grid_results:
            log.info('{!s}, {!s}, {!s}'.format(g.set, g.clf, self.grid_results.index(g)))

    def get(self, set, clf):
        for g in self.grid_results:
            if g.set == set and clf == g.clf:
                return g

    def print_top_n(self, n, clf=None):
        """
        Prints top highest rows.
        :param n: # top rows
        :param clf: only for a certain clf 'svm' or 'rf'
        :return: void
        """
        res = []
        for g in self.grid_results:
            if g.clf == clf:
                res.append(g.print_top_n(n))
            else:
                res.append(g.print_top_n(n))
        return res

    def write_top_n(self, n):
        pandas.set_option('display.max_colwidth', -1)

        for g in self.grid_results:
            fname = settings.ANALYSIS_FOLDER + '/gridcv_{!s}_{!s}.html'.format(g.clf, g.set)
            with open(fname, 'w') as f:
                f.write(g.print_top_n(n=n).to_html())
        pandas.reset_option('display.max_colwidth')

    def print_best(self):

        full_str = '\n\n'
        for g in self.grid_results:
            best_str = 'Set: {!s}\t\tGridsize: {!s}\t\tScore: {!s}\t\tParams: {!s}'.format(g.set, len(
                g.grid_cv.cv_results_['mean_test_score']), g.grid_cv.best_score_, g.grid_cv.best_params_)
            full_str += best_str + '\n'

        log.info('Printing summary of best parameters per set: {!s}'.format(full_str))
    def print_top_n_deprecated(self, n=10, reverse=False):
        topn_string = ''
        for g in self.grid_results:
            if len(g.gridcv.cv_results_['mean_test_score']) >= n:
                if reverse:
                    topn_string += '\n Last {!s}, set {!s}, gridsize {!s}'.format(n, g.set,
                                                                                  len(g.gridcv.cv_results_['mean_test_score']))
                else:
                    topn_string += '\n Top {!s}, set {!s}, gridsize {!s}'.format(n, g.set,
                                                                                 len(g.gridcv.cv_results_['mean_test_score']))

                a = g.gridcv.cv_results_['mean_test_score']
                indices = sorted(range(len(a)), key=lambda i: a[i], reverse=reverse)[-n:]
                scores = numpy.array(g.gridcv.cv_results_['mean_test_score'])[indices]
                params = numpy.array(g.gridcv.cv_results_['params'])[indices]
                mean_fit_time = numpy.array(g.gridcv.cv_results_['mean_fit_time'])[indices]
                for i in range(len(indices)):
                    topn_string += '\nScore: {!s}\t\tParams: {!s} \t\t MeanFitTime: {!s}'.format(scores[i], params[i],
                                                                                                 mean_fit_time[i])

            topn_string += '\n###########################################################\n'
        log.info(topn_string)


def open_pickle_file(file: str):
    """
    Open a pickle file to be able to view results again.
    :param file:
    :return:
    """
    with open(file, 'rb') as f:
        unpickled = pickle.load(f)
        return unpickled


def stats_list_filter_high_fnr(stats):
    for s in stats:
        if len(s.missclassified) > 100:
            print(random.sample(s.missclassified, k=10))


class SingleRunStatistic:
    """
    Needed datastructure for temp. storing results during parallel cv.
    """
    def __init__(self, y_true, y_pred, domains_test):
        self.y_true = y_true
        self.y_pred = y_pred
        self.domain_test = domains_test


class MixVsXStatistic:

    def __init__(self, file):
        self.res, self.dmns = open_pickle_file(file)

    def print_all(self):

        for r in self.res:
            b_b, b_dga1, b_notdga1, lbls, m_b, m_dga1, m_notdga1 = self.count_cms(r)
            res_string = self.print_cms(b_b, b_dga1, b_notdga1, lbls, m_b, m_dga1, m_notdga1, r)
            print(res_string)

    def print_cms(self, b_b, b_dga1, b_notdga1, lbls, m_b, m_dga1, m_notdga1, r):
        res_string = ''
        res_string += '{!s}, {!s}, {!s}\n'.format(r[0], r[1], len(lbls))
        res_string += '\t Benign \t DGA1 \t not_DGA1\n'
        res_string += 'Benign \t {!s} \t {!s}\t {!s}\n'.format(b_b, b_dga1, b_notdga1)
        res_string += 'Malic. \t {!s} \t {!s}\t {!s}\n'.format(m_b, m_dga1, m_notdga1)
        return res_string

    def export_txt(self):
        with open(settings.ANALYSIS_FOLDER + '/mix_vs_x.txt', 'a+') as f:
            for r in self.res:
                b_b, b_dga1, b_notdga1, lbls, m_b, m_dga1, m_notdga1 = self.count_cms(r)
                res_string = self.print_cms(b_b, b_dga1, b_notdga1, lbls, m_b, m_dga1, m_notdga1, r)
                f.write(res_string + '\n')

    def count_cms(self, r):
        b_b = 0
        b_dga1 = 0
        b_notdga1 = 0
        m_b = 0
        m_dga1 = 0
        m_notdga1 = 0
        lbls = r[2]
        pred_lbls = r[3]
        for i in range(len(lbls)):
            if pred_lbls[i] == 0:
                if lbls[i] == 0:
                    b_b += 1
                elif lbls[i] == 1:
                    b_dga1 += 1
                elif lbls[i] == 2:
                    b_notdga1 += 1

            elif pred_lbls[i] == 1:
                if lbls[i] == 0:
                    m_b += 1
                elif lbls[i] == 1:
                    m_dga1 += 1
                elif lbls[i] == 2:
                    m_notdga1 += 1
        return b_b, b_dga1, b_notdga1, lbls, m_b, m_dga1, m_notdga1

    def union_of_false_positives(self):
        fps = set()
        for r in self.res:
            lbls = r[2]
            pred_lbls = r[3]
            for i in range(len(lbls)):
                if pred_lbls[i] == 1 and lbls[i] == 0:
                    fps.add(self.dmns[i])

        print(str(fps))
        print(str(len(fps)))

    def intersection_of_false_positives(self):
        fps = []
        for r in self.res:
            lbls = r[2]
            pred_lbls = r[3]
            fp = set()
            for i in range(len(lbls)):
                if pred_lbls[i] == 1 and lbls[i] == 0:
                    fp.add(self.dmns[i])
            fps.append(fp)

        intersection = set.intersection(*fps)

        print(str(intersection))
        print(str(len(intersection)))


def analyze_eval_file(file_path_from):
    full_dict = open_pickle_file(file_path_from)
    joined_stats = dict()

    for k in full_dict.keys():
        joined_stats[k] = Statistic(set_id=full_dict[k][0].set_id, id=full_dict[k][0].id)
        for stats in full_dict[k]:
            joined_stats[k].add_from_self(stats)
            log.debug(joined_stats[k].scores)

        joined_stats[k] = joined_stats[k].mean_scores()

    df = DataFrame(joined_stats)
    return df


class ResultWorkspace:
    """
    Represents result files from cv and logo evaluations.
    """
    def __init__(self, file=None):

        self.dfs = []

        if file:
            self.dfs.append(MetaStatistic(file).df)
        else:

            for (path, _, files) in walk(settings.EVAL_FOLDER):
                eval_files = files
                eval_files = [path + '/' + f for f in eval_files]
                break
            for f in eval_files:
                self.dfs.append(MetaStatistic(f).df)

        self.df = pandas.concat(self.dfs)

    def write_to_html(self):
        pandas.set_option('display.max_colwidth', -1)
        header = '{!s}'.format(self.df.index.tolist()[0])
        df = self.df.reset_index(level=['Clf.', 'Set_Type', 'Eval.'])
        if '#Rep.' in df:
            df.drop('#Rep.', 1, inplace=True)

        df.drop('Eval.', 1, inplace=True)
        df.drop('Set_Size', 1, inplace=True)
        df.drop('Set_Type', 1, inplace=True)
        df.drop('f1', 1, inplace=True)
        df.drop('precision', 1, inplace=True)
        df.columns = ['Clf', '\\ac{DGA} Type', '\\ac{ACC}', '\\ac{TPR}', '\\ac{TNR}', '\\ac{FNR}', '\\ac{FPR}']
        fname = settings.ANALYSIS_FOLDER + '/eval_full.html'
        with open(fname, 'w') as f:
            f.write(df.to_html())
        pandas.reset_option('display.max_colwidth')

    def write_to_tex(self):
        pandas.set_option('display.max_colwidth', -1)

        fname = settings.ANALYSIS_FOLDER + '/eval_full.tex'
        with open(fname, 'w') as f:
            f.write(self.df.to_latex())
        pandas.reset_option('display.max_colwidth')

    def summarizing_tables_tex(self):
        for df in self.dfs:
            print('{!s}'.format(df.index.tolist()[0]))
            desc = df.describe()
            if '#Rep.' in desc:
                desc.drop('#Rep.', 1, inplace=True)
            if 'DGA_Type' in desc:
                desc.drop('DGA_Type', 1, inplace=True)
            if 'Set_Size' in desc:
                desc.drop('Set_Size', 1, inplace=True)
            desc.drop('f1', 1, inplace=True)
            desc.drop('precision', 1, inplace=True)
            desc.columns = ['\\ac{ACC}', '\\ac{TPR}', '\\ac{TNR}', '\\ac{FNR}', '\\ac{FPR}']
            desc = desc.round(5)
            latex_str = desc.to_latex()
            print(latex_str)

    def long_tables_tex(self):
        for df in self.dfs:
            header = '{!s}'.format(df.index.tolist()[0])
            df = df.reset_index(level=['Clf.', 'Set_Type', 'Eval.'])
            if '#Rep.' in df:
                df.drop('#Rep.', 1, inplace=True)

            df.drop('Eval.', 1, inplace=True)
            df.drop('Set_Size', 1, inplace=True)
            df.drop('Clf.', 1, inplace=True)
            df.drop('Set_Type', 1, inplace=True)
            df.drop('f1', 1, inplace=True)
            df.drop('precision', 1, inplace=True)
            df.columns = ['\\ac{DGA} Type', '\\ac{ACC}', '\\ac{TPR}', '\\ac{TNR}', '\\ac{FNR}', '\\ac{FPR}']
            df = df.round(5)
            df.sort_index(inplace=True)
            latex_str = df.to_latex(index=True)
            latex_str = latex_str.title()
            latex_str = latex_str.replace('Tdd', '\\ac{TDD}')
            latex_str = latex_str.replace('Tdn', '\\ac{TDN}')
            latex_str = latex_str.replace('Tid', '\\ac{TID}')
            latex_str = latex_str.replace('Tin', '\\ac{TIN}')

            latex_str = latex_str.replace('Dga', 'DGA')
            latex_str = latex_str.replace('Tpr', 'TPR')
            latex_str = latex_str.replace('Tnr', 'TNR')
            latex_str = latex_str.replace('Fnr', 'FNR')
            latex_str = latex_str.replace('Fpr', 'FPR')
            latex_str = latex_str.replace('Acc', 'ACC')

            latex_str = latex_str.replace('Toprule', 'toprule')
            latex_str = latex_str.replace('Midrule', 'midrule')
            latex_str = latex_str.replace('Bottomrule', 'bottomrule')

            latex_str = latex_str.replace('Begin', 'begin')
            latex_str = latex_str.replace('End', 'end')
            latex_str = latex_str.replace('Tabular', 'tabular')
            latex_str = latex_str.replace('Lllrrrrr', 'lllrrrrr')

            with open(settings.ANALYSIS_FOLDER + '/full_res_tables.txt', 'a+') as f:
                f.write(header)
                f.write(latex_str)


class MetaStatistic:
    """
    Encapsulation and presentation of multiple Statistics, which results from kfold or logo analysis.
    """
    def __init__(self, file_path_from):
        """
        Load all statistics serialized in the folder given as parameter.
        :param file_path_from: from where to import.
        """
        self.full_stats = open_pickle_file(file_path_from)
        self.joined_stats = dict()
        for k in self.full_stats.keys():
            log.debug('Joining stats for key {!s}'.format(k))
            self.joined_stats[k] = self._stats_joiner(self.full_stats[k])

        self.df = self._to_pd_df()

    @staticmethod
    def _stats_joiner(list_of_stats):
        if not list_of_stats:
            return None

        new_stats = Statistic(set_id=list_of_stats[0].set_id, id=list_of_stats[-1].id, desc=list_of_stats[0].desc)

        for st in list_of_stats:
            new_stats.add_from_self(st)

        return new_stats

    @staticmethod
    def _meta_info_extraction(st):

        set_id = st.set_id

        if 'svm' in st.id:
            clf_type = 'svm'
        elif 'rf' in st.id:
            clf_type = 'rf'

        set_count = set_id.split('_')[-1]

        eval = st.id.split('_')[0]

        if 'single' in set_id:
            dga_family = set_id.split('_')[-2]

            dga_type = classifiers.DGA_TYPE_MAP[dga_family]

            if settings.SetTypes.single_dga_grouped_day.value in set_id:
                set_type = settings.SetTypes.single_dga_grouped_day.value
                size = str(int(st.set_id.split(settings.SetTypes.single_dga_grouped_day.value)[1].split('_')[1]) * 2)
            elif settings.SetTypes.single_dga_grouped_seed.value in set_id:
                set_type = settings.SetTypes.single_dga_grouped_seed.value
                size = str(int(st.set_id.split(settings.SetTypes.single_dga_grouped_seed.value)[1].split('_')[1]) * 2)
            else:
                set_type = settings.SetTypes.single_dga.value
                size = str(int(st.set_id.split(settings.SetTypes.single_dga.value)[1].split('_')[1]) * 2)
        elif 'mix' in set_id:
            dga_family = 'mix'
            dga_type = 'mix'
            set_type = settings.SetTypes.mixed_dga_grouped_family.value
            size = 50000 # TODO
            # size = str(int(st.set_id.split(settings.SetTypes.mixed_dga_grouped_family.value)[1].split('_')[1]) * 2)

        else:
            # TODO
            dga_family = set_id.split('_')[-2]
            dga_type = classifiers.DGA_TYPE_MAP[dga_family]
            set_type = settings.SetTypes.single_dga.value
            size = 50000

        if '_rep' in st.id:
            repetitions = str(int(st.id.split('_rep', 1)[1].split('_')[0]) + 1)
        else:
            repetitions = 1

        return clf_type, eval, dga_type, repetitions, dga_family, set_type, size, set_count

    def _to_pd_df(self):

        # corr, describe, 

        index = ['Clf.', 'Set_Type', 'DGA_Family', 'Set_#', 'Eval.']
        columns = ['Set_Size', 'DGA_Type', '#Rep.', 'accuracy', 'recall', 'tnr', 'precision', 'fnr', 'fpr', 'f1']
        scoring = ['accuracy', 'recall', 'tnr', 'precision']

        df_dict = defaultdict(list)
        log.debug('Generating pandas DataFrame.')

        for k in self.joined_stats.keys():
            log.debug('Adding {!s} to the dataframe.'.format(k))
            st = self.joined_stats[k]

            clf_type, eval, dga_type, repetitions, dga_family, set_type, size, set_count = self._meta_info_extraction(st)
            df_dict['DGA_Family'].append(dga_family)
            df_dict['Set_Type'].append(set_type)
            df_dict['Set_#'].append(set_count)
            df_dict['Set_Size'].append(size)
            df_dict['DGA_Type'].append(dga_type)
            df_dict['Eval.'].append(eval)
            df_dict['Clf.'].append(clf_type)
            df_dict['#Rep.'].append(repetitions)

            cum_cm = st.cummulative_conf_matrix()

            df_dict['fpr'].append(cum_cm[0,1])
            df_dict['tnr'].append(cum_cm[0,0])
            df_dict['fnr'].append(cum_cm[1,0])

            st_means = st.mean_scores()
            for l in st_means.keys():
                df_dict[l].append(st_means[l])

        df = pandas.DataFrame(df_dict)
        df.set_index(index, inplace=True)
        df.pop('roc')
        # df.sort_values(scoring, inplace=True, ascending=False)
        df = df[columns]
        return df

    def eval_fp_reduction(self, alexa_count=1000000, only_global=False, siemens=False):

        fps = []
        log.info('Collecting false positives')
        for k in self.joined_stats.keys():
            for m in self.joined_stats[k].missclassified:
                if m[1] == 0:
                    fps.append(m[0])

        fps = list(set(fps))

        log.info('FP Count: {!s}'.format(len(fps)))
        log.info('Starting reduction by alexa top 1m domains...')

        start = len(fps)
        fps = data.fp_reduction_alexa_top1m(fps, alexa_count)
        alexa_absolute = start - len(fps)
        alexa_percentage = 100/start * alexa_absolute
        after_alexa = start - alexa_absolute

        if not only_global:

            log.info('Starting reduction by local domains...')

            fps = data.fp_reduction_local_domains(fps, siemens=siemens)

            log.info('FP Count: {!s}'.format(len(fps)))
            local_absolute = after_alexa - len(fps)
            local_percentage = 100/after_alexa * local_absolute

            overall_absolute = start - len(fps)
            overall_percentage = 100/start * overall_absolute

        log.info('Reduction Results\n\n Start {!s} \n Finish {!s} \n\n Alexa: {!s} ({!s}) \n Local: {!s} ({!s}) \n Overall: {!s}'
                 ' ({!s}) '.format(start, start-overall_absolute, alexa_absolute, alexa_percentage, local_absolute, local_percentage, overall_absolute, overall_percentage))
        return fps

class Statistic:
    """
    Encapsulation to hold results for multiple runs and automatically cumulate them (over the same set).
    """

    def __init__(self, set_id= None, id=None, desc=None):
        if id:
            self.id = id
        else:
            self.id = ''.join(get_rand_id())

        self.set_id = set_id
        self.desc = desc
        self.cms = []
        self.ys = []
        self.y_true = []
        self.y_pred = []
        self.missclassified = []
        self.scores = defaultdict(list)

    def add_run_from_single(self, single_run: SingleRunStatistic):
        self.add_run(single_run.y_true, single_run.y_pred, single_run.domain_test)

    def add_from_self(self, s):
        self.cms.extend(s.cms)
        self.ys.extend(s.ys)
        self.y_true.extend(s.y_true)
        self.y_pred.extend(s.y_pred)
        self.missclassified.extend(s.missclassified)
        for k in s.scores.keys():
            self.scores[k] += s.scores[k]

    def add_run(self, y_true, y_pred, domains_test):
        """
        Add a completed run
        :param domains_test:
        :param y_true: true labels
        :param y_pred:  predicted labels
        :return:
        """
        log.verbose('Adding run.\ny_true: {!s}\ny_pred: {!s}'.format(y_true, y_pred))
        self.ys.append((y_true, y_pred))
        self.y_true = numpy.concatenate((self.y_true, y_true))
        self.y_pred = numpy.concatenate((self.y_pred, y_pred))
        self.cms.append(confusion_matrix(y_true, y_pred))
        self.scores = {'accuracy': [accuracy_score(y_true, y_pred)],
                       'precision': [precision_score(y_true, y_pred)],
                       'recall': [recall_score(y_true, y_pred)],
                       'roc': [roc_auc_score(y_true, y_pred)],
                       'f1': [f1_score(y_true, y_pred)]}
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                self.missclassified.append((domains_test[i], y_true[i]))

    def mean_scores(self):
        mean_scores = dict()
        for k in self.scores.keys():
            mean_scores[k] = sum(self.scores[k]) / len(self.scores[k])
        return mean_scores

    def serialize(self, where=None, keep_copy=True):

        if not where:
            stats_file = settings.EVAL_FOLDER + self.id
        else:
            stats_file = where + self.id
        data.serialize_keep_copy(stats_file, self, keep_copy=keep_copy)

    @staticmethod
    def load(file: str):
        """
        Load statistics from disk.
        :param file: to load
        :return: deserialized Statistics object.
        """
        with open(file, 'rb') as f:
            stats = pickle.load(f)

        return stats

    def print_cummulative_clf_report(self):
        """
        Prints the cumulated classification report for all added runs
        :return:
        """
        log.info('Classification report:\n\n' + classification_report(self.y_true, self.y_pred,
                                                                      target_names=['Benign', 'Malicious']))

    def cummulative_conf_matrix(self, in_percent=True):
        cumu_cm = numpy.zeros(shape=(2, 2), dtype=int)

        for cm in self.cms:
            cumu_cm = numpy.add(cumu_cm, cm)

        if in_percent:
            cumu_cm = cumu_cm / cumu_cm.astype(numpy.float).sum(axis=1)

        return cumu_cm

    def print_cummulative_conf_matrix(self):
        """
        Prints the cumulated confusion matrix for all added runs
        :return:
        """
        cumu_cm = self.cummulative_conf_matrix()
        log.info('Confusion matrix\n' + pretty_print_cm(cumu_cm.astype(int), ('Benign', 'Malicious')))

    def print_full_summary(self):

        self.print_cummulative_clf_report()
        self.print_cummulative_conf_matrix()
        log.info('Missclassifications <domain, true_lbl>:\n{!s}'.format(str(self.missclassified)))


def pretty_print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    Pretty printer for the confusion matrix of sklearn
    :param cm: confusion matrix
    :param labels: labels for the classes
    :param hide_zeroes:
    :param hide_diagonal:
    :param hide_threshold:
    :return: pretty string
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = ' ' * columnwidth
    # Print header
    pretty_cm_str = '    ' + empty_cell + ' '
    for label in labels:
        pretty_cm_str += '%{0}s'.format(columnwidth) % label + ' '
    pretty_cm_str += '\n'
    # Print rows
    for i, label1 in enumerate(labels):
        pretty_cm_str = pretty_cm_str + '    %{0}s'.format(columnwidth) % label1 + ' '
        for j in range(len(labels)):
            cell = '%{0}.1f'.format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            pretty_cm_str += cell + ' '
        pretty_cm_str += '\n'

    return pretty_cm_str


def _get_scores(estimator, x, y):
    """
    Helper for sklearn, required to be able to score by more than one metric.
    :param estimator:
    :param x:
    :param y:
    :return:
    """
    y_pred = estimator.predict(x)
    return (accuracy_score(y, y_pred),
            precision_score(y, y_pred),
            recall_score(y, y_pred),
            roc_auc_score(y, y_pred),
            f1_score(y, y_pred))


def multi_scorer_gridsearch(estimator, x, y):
    """
    Helper for sklearn, required to be able to score by more than one metric.
    :param estimator:
    :param x:
    :param y:
    :return:
    """
    with __lock:
        if isinstance(estimator, SVC):
            if estimator.kernel == 'rbf':
                params = {'C': estimator.C, 'gamma': estimator.gamma, 'kernel': estimator.kernel}
            else:
                params = {'C': estimator.C, 'kernel': estimator.kernel}
        if isinstance(estimator, RandomForestClassifier):
            log.info('RandomForestClassifier')
            params = {'max_features': estimator.max_features, 'n_estimators': estimator.n_estimators,
                      'criterion': estimator.criterion}

        try:
            with open(settings.GRID_SEARCH_FOLDER_TMP + ''.join(__rand_id), 'rb') as f:
                all_scores = pickle.load(f)
        except FileNotFoundError:
            all_scores = dict()

        accuracy, precision, recall, roc, f1 = _get_scores(estimator, x, y)

        scores = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'roc': roc, 'f1': f1}

        params = frozenset(params.items())
        if params not in all_scores.keys():
            all_scores[params] = []

        all_scores[params].append(scores)
        with open(settings.GRID_SEARCH_FOLDER_TMP + ''.join(__rand_id), 'wb+') as f:
            pickle.dump(all_scores, f)

    return roc


class PcapClassificationStats:
    def __init__(self, file):
        self.results = dict()
        with open(settings.ANALYSIS_FOLDER + '/' + file, 'r') as f:
            result_list = json.load(f)

        for pcap, res in result_list:
            self.results = {**self.results, **res}

    def malicious_count(self):
        dom_count = 0
        pred_count = 0
        mal_pred_count = 0
        mal_domain_count = 0

        for res in self.results:
            pass
