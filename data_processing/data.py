import json
import logging
import math
import os
import pickle
import random
import string
import subprocess
from collections import defaultdict
from os import walk, path
from shutil import copy2

import numpy

import settings
from data_processing import dgarchive_api
from settings import SCRIPTS_ROOT, WORKSPACE_ROOT, PCAPS_FOLDER, DGARCHIVE_FODLER, SETS_FOLDER

log = logging.getLogger('log')


def clean_domain_list(domain_list: list, dga=False):
    """
    Cleans a given domain list from invalid domains and cleans each single domain in the list.
    :param domain_list:
    :param dga:
    :return:
    """

    domain_list = [d.strip().lower() for d in domain_list]
    domain_list = list(filter(None, domain_list))

    if dga:
        # some ramnit domains ending with the pattern: [u'.bid', u'.eu']
        to_remove = []
        for d in domain_list:
            if '[' in d:
                log.verbose('Domain contains [: {!s}'.format(d))
                to_remove.append(d)
                res = set()
                bracket_split = d.split('[')
                tlds = bracket_split[1].split(',')
                for tld in tlds:
                    tld = tld.strip()
                    tld = tld.split("'")[1].replace('.', '')
                    res_d = bracket_split[0] + tld
                    res.add(res_d)
                    log.verbose('Cleaned domain: {!s}'.format(res_d))
                    domain_list.append(res_d)

        domain_list = [d for d in domain_list if d not in to_remove]

    return domain_list


def filter_non_valid(domains):
    log.info('Filtering invalid domains...')
    valid_tlds = PublicSuffixes().get_valid_tlds()
    return {d for d in domains if '.' + d.split('.')[-1] in valid_tlds}


def fp_reduction_local_domains(domains, siemens=False, return_removed=False):

    if not siemens:
        local_filter_set = settings.LOCAL_FILTER_SET_RWTH
    else:
        local_filter_set = settings.LOCAL_FILTER_SET_SIEMENS

    res = set(domains)
    removed = set()
    for d in domains:
        for suf in local_filter_set:
            if d.endswith(suf):
                res.discard(d)
                removed.add(d)

    if return_removed:
        return removed
    else:
        return res

def fp_reduction_alexa_top1m(domains, count, return_removed=False):

    alexatop = AlexaTopX(x=count).data
    res = set(domains)
    removed = set()
    for d in domains:
        for suf in alexatop:
            if d.endswith(suf):
                res.discard(d)
                removed.add(d)
    if return_removed:
        return removed
    else:
        return res


class LabeledData:
    """
    Structure to hold a domain name as string and its corresponding label. (For ease of use)
    """
    def __init__(self, domain: str, malicious: bool):
        """

        :param domain: domain name as string
        :param malicious: true label, True if malicious False if not
        """
        self.domain = domain.lower()
        self.malicious = malicious

    def human_readable_label(self):
        """

        :return: a human readable label instead of 1 or 0
        """
        if self.malicious:
            return 'malicious'
        else:
            return 'benign'

    def __str__(self):
        return self.domain + ', ' + self.human_readable_label()

    def __repr__(self):
        return self.__str__()


class DataSet:
    """
    Class providing abstract view on a set of LabeledData. Two classes: benign and malicious
    """
    def __init__(self, benign, malicious, families, id=None):
        """

        :param benign: iterable containing LabeledData
        :param malicious: iterable containing LabeledData
        :param id: a name, mainly used as file name for serialization. A random name is choosen if none is given.
        """
        if id:
            self.id = id
        else:
            self.id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        self.families = families
        self.benign = list(benign)
        self.malicious = list(malicious)
        self.full = []
        self.full += self.benign
        self.full += self.malicious

    def serialize(self, keep_copy=True, where=None):
        """
        Save data set to disk.
        :param keep_copy: if the file exists this one as a copy. If False nothing is stored to disk.
        :param where: path to location to store set file
        :return: void
        """
        if not where:
            set_file = SETS_FOLDER + '{!s}.pkl'.format(self.id)
        else:
            set_file = where + '{!s}.pkl'.format(self.id)

        serialize_keep_copy(set_file, self, keep_copy=keep_copy)

    def expand(self):
        """
        Expands iterables of LabeledData to two numpy arrays, one containing the data and one the labels
        This is necessary for the work with sklearn.
        :return: domain names as numpy arrays, labels as numpy arrays
        """
        domains = []
        labels = []

        for d in self.full:
            domains.append(d.domain)
            labels.append(d.malicious)

        domains = clean_domain_list(domains, dga=False)

        return numpy.array(domains), numpy.array(labels)

    @staticmethod
    def load(file: str):
        """
        Load data set from disk.
        :param file: to load
        :return: deserialized DataSet object.
        """
        with open(file, 'rb') as f:
            dataset = pickle.load(f)

        return dataset


def serialize_keep_copy(where, what, keep_copy=True):
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
        pickle.dump(what, f)


class GroupedDataSet(DataSet):
    """
    Dataset additionally holding integer labels to indicate a group.
    This is used for working with LeaveKGroupOut CrossValidation.
    """
    def __init__(self, benign, malicious, groups, group_map, id=None, type=None):
        super().__init__(benign, malicious, list(group_map.values()), id=id)
        self.groups = groups + groups
        self.family_map = group_map
        self.type = type

    def expand(self):
        """
        Return domain, labels and the group labels as numpy arrays.
        :return: 3 numpy arrays
        """
        dmns, lbls = super().expand()
        return dmns, lbls, numpy.array(self.groups)


class Workspace:
    """
    Encapsulates DGArchive data and data from the pcap files.
    It is assumed that all pcaps imported represent benign data.
    """
    def __init__(self, days=10, empty=False, consider_tuples=False, no_import=False):
        """

        :param days: #days to consider from DGArchive data. Be careful, this may consume much RAM when set too high.
        Better import in several steps using the extend_dga_data method.
        :param empty: set this if you only want to work with serialized sets and do not need any fresh ones.
        """
        log.info('Setting up workspace. Considering {!s} random days of DGArchive.'.format(days))
        if not empty:
            self.dga_data = DGArchive(days=days, no_import=no_import)
            if not no_import:
                self.pcaps = BenignData(consider_tuples=consider_tuples)
        else:
            self.dga_data = None
            self.pcaps = None

        self.data_sets_loaded = dict()
        self.data_sets_cache = defaultdict(list)
        for (path, dirs, files) in walk(SETS_FOLDER):
            self.data_sets_cache[''].extend(files)
            for d in dirs:
                for (_, _, files) in walk(path + d):
                    self.data_sets_cache[d].extend(files)
                    break
            break

        for k in self.data_sets_cache.keys():
            self.data_sets_cache[k] = [SETS_FOLDER + '/' + k + '/' + f for f in self.data_sets_cache[k]]


    def families_to_disjoint_days(self):
        """
        helper method for statistics
        Counts disjoint days per DGA
        :return:
        """
        self.dga_data.families = ['chinad', 'conficker', 'corebot', 'cryptolocker', 'dyre', 'gameover', 'gozi', 'locky', 'murofet', 'murofetweekly', 'necurs', 'nymaim', 'oderoor', 'pushdo', 'qadars', 'qakbot', 'ranbyus', 'suppobox', 'symmi', 'vidro', 'virut']
        for family in self.dga_data.families:
            day_map = self.dga_data.lazy_load_daily_seed_map(family)
            backlog = []
            disjoint_days = []
            for k in day_map.keys():
                if not self._similar(day_map[k], backlog):
                    disjoint_days.append(k)
                    backlog.append(day_map[k])
            if len(disjoint_days) > 4:
                print('{!s} & {!s} \\\\'.format(family.capitalize(), len(disjoint_days)))

    def serialize(self):
        """
        Save current workspace to disk.
        :return:
        """
        log.info('Storing workspace to disk. This may take a while...')
        serialize_keep_copy(settings.WORKSPACE_FILE, self.__dict__, keep_copy=False)
        log.info('Finished storing')

    def load_from_disk(self):
        """
        Load a workspace from disk.
        :return:
        """
        log.info('Loading workspace from disk. This may take a while...')
        with open(settings.WORKSPACE_FILE, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        log.info('Finished loading')

    def load(self, dataset_id, subfolder=''):
        """
        Load the given dataset from disk. All available datasets are listed in data_sets_cache
        The loaded data set is added to the data_sets_loaded list.
        :param dataset_id: id to load
        :return: void
        """
        to_load = ''
        for d in self.data_sets_cache[subfolder]:
            if dataset_id in d:
                to_load = d
                break
        log.info('Loading data set: {!s}'.format(dataset_id))

        self.data_sets_loaded[dataset_id] = DataSet.load(to_load)

    def random_set(self, size=1000, families=None, nx=True):
        """
        Generates a random data set containing as much benign samples as malicious ones.
        :param size: size of one class, i.e., the overall size will be doubled
        :param families: list of strings denoting DGA family names. For valid family names check out DGA json files.
        If None all families are considered.
        :return:
        """
        # this method yields cleaned domain lists
        malicious = self.dga_data.random_sample(size=size, families=families)
        benign = self.pcaps.random_sample(k=len(malicious), nx=nx)
        dataset = DataSet(benign, malicious, families)
        self.data_sets_loaded[dataset.id] = dataset
        return dataset

    @staticmethod
    def _similar(domain_set, list_of_domain_sets):
        """
        Compares two sets of domains and determines whether they are similar or not.
        :param domain_set:
        :param list_of_domain_sets:
        :return: False if at least 75 percent uniquness is present in each set.
        """
        for s in list_of_domain_sets:
            sym_dif = s ^ domain_set # non-common elements
            # at least 75% uniquness
            if len(sym_dif) < len(domain_set) * 0.75:
                return True
        return False

    def grouped_daily_random_set(self, family, size=1000, max_group_count=64):
        """
        Generates a set grouped by day.
        :param family:
        :param size:
        :param max_group_count:
        :return:
        """
        log.info('Gernerating single grouped day dga sets for: {!s}.'.format(family))
        day_map = self.dga_data.lazy_load_daily_seed_map(family)
        backlog = []
        disjoint_days = []
        min_size = 250
        for k in day_map.keys():
            if not self._similar(day_map[k], backlog):
                disjoint_days.append(k)
                backlog.append(day_map[k])

        if len(disjoint_days) < 5:
            raise ValueError('For the family {!s} not enough disjoint days exist.'.format(family))

        log.debug('Disjoint days: {!s}'.format(disjoint_days))
        log.debug('Disjoint sets count {!s}'.format(len(disjoint_days)))

        merged_day_map = defaultdict(set)

        if len(disjoint_days) > max_group_count:
            disjoint_days = random.sample(disjoint_days, max_group_count)

        overall_size = 0
        for k in disjoint_days:
            merged_day_map[k] = day_map[k]
            overall_size += len(day_map[k])

        if overall_size < min_size:
            raise ValueError('For the family {!s} not enough disjoint domains exist.'.format(family))

        sample_size = math.floor(size / len(merged_day_map.keys()))
        group_map = dict()
        malicious = []
        groups = []
        i = 0
        for k in merged_day_map.keys():
            group_map[i] = k
            if len(merged_day_map[k]) < sample_size:
                log.debug('Sample size too small: {!s} < {!s}'.format(len(merged_day_map[k]), sample_size))
                cur_sample = merged_day_map[k]
            else:
                cur_sample = random.sample(merged_day_map[k], sample_size)
            malicious.extend(cur_sample)
            groups += [i] * len(cur_sample)
            i += 1

        malicious = [LabeledData(d, 1) for d in malicious]

        benign = self.pcaps.random_sample(k=len(malicious))
        dataset = GroupedDataSet(benign, malicious, groups, group_map, type=settings.SetTypes.single_dga_grouped_day.value)
        self.data_sets_loaded[dataset.id] = dataset
        return dataset

    def grouped_random_set(self, size=1000, families=None):
        """
        If families contain exactly one family, a leave out seed set is constructed.
        Else a leave out family set is constructed.
        The sets are not necessarily exactly qsame sized. This may depend on the availability of the amount of DGA domains for certain families.
        :param size: size of one class, i.e., the overall size will be doubled
        :param families: If exactly one family is passed this family is grouped by seeds. If None, all families are considered.
        :return: data set according to the params given
        """
        # this method yields cleaned domain lists

        groups = []
        malicious = []
        if not families:
            families = self.dga_data.families

        group_map = dict()
        if len(families) == 1:
            type = settings.SetTypes.single_dga_grouped_seed.value
            seed_map = self.dga_data.lazy_load_family_seed_map(families[0])
            seed_count = len(seed_map.keys())
            log.debug('seed count: {!s}'.format(seed_count))

            if not seed_count > 1:
                raise ValueError('For the given family no multiple seeds are known.')

            sample_size = math.floor(size / seed_count)
            i = 0
            for s in seed_map.keys():
                group_map[i] = s
                if len(seed_map[s]) < sample_size:
                    log.debug('Sample size too small: {!s} < {!s}'.format(len(seed_map[s]), sample_size))
                    cur_sample = seed_map[s]
                else:
                    cur_sample = random.sample(seed_map[s], sample_size)
                malicious.extend(cur_sample)
                groups += [i] * len(cur_sample)
                i += 1
            malicious = [LabeledData(d, 1) for d in malicious]
        else:
            type = settings.SetTypes.mixed_dga_grouped_family.value
            sample_size = math.floor(size / len(families))
            i = 0
            for f in families:
                sample = self.dga_data.random_sample(size=sample_size, families=[f], fill=False)
                group_map[i] = f
                malicious += sample
                groups += [i] * len(sample)
                i += 1

        benign = self.pcaps.random_sample(k=len(malicious))
        dataset = GroupedDataSet(benign, malicious, groups, group_map, type=type)
        self.data_sets_loaded[dataset.id] = dataset
        return dataset

    def load_all(self, which=''):
        """
        Loads all cached datasets. They are all available in the data_sets_loaded list afterwards.
        :return:
        """
        for d in self.data_sets_cache[which]:
            self.load(os.path.basename(d), subfolder=which)

    def extend_dga_data_by_n_days(self, days):
        """
        Import further DGArchive days to the workspace.
        :param days: no. of days to additionally consider
        :return:
        """
        self.dga_data.extend_dga_data(days)

    def stats(self):
        """
        Print basic statistics about the workspace.
        :return:
        """
        log.info('Cached data sets: {!s}'.format(self.data_sets_cache))
        log.info('Loaded data sets: {!s}'.format(self.data_sets_loaded.keys()))

    def gen_single_dga_set(self, families=None, size=20000, count=1):
        """
        Generate sets with one DGA each.
        :param families: list of families to consider
        :param size: set size
        :param count: how many sets per family
        :return:
        """
        if not families:
            families = self.dga_data.families

        for fam in families:
            log.info('Gernerating single dga set for: {!s}.'.format(fam))
            for i in range(count):
                cur_set = self.random_set(size=size, families=[fam])
                cur_set.id = '{!s}_{!s}_{!s}_{!s}'.format(settings.SetTypes.single_dga.value, size, fam, i)
                cur_set.serialize(where=settings.SETS_FOLDER + settings.SetTypes.single_dga.value + '/')
                self.data_sets_loaded.clear()

    def gen_single_grouped_seed_set(self, families=None, size=20000, count=1):
        """
        Generate grouped sets with one DGA each, grouped by seeds.
        :param families: list of families to consider
        :param size: set size
        :param count: how many sets per family
        :return:
        """
        if not families:
            families = self.dga_data.families

        for fam in families:
            log.info('Gernerating single grouped seeded dga sets for: {!s}.'.format(fam))

            try:
                for i in range(count):
                    cur_set = self.grouped_random_set(size=size, families=[fam])
                    cur_set.id = '{!s}_{!s}_{!s}_{!s}'.format(settings.SetTypes.single_dga_grouped_seed.value, size, fam, i)
                    cur_set.serialize(where=settings.SETS_FOLDER + settings.SetTypes.single_dga_grouped_seed.value + '/')
                    self.data_sets_loaded.clear()
            except ValueError:
                log.info('Skipping family {!s}. Only 1 seed known.'.format(fam))

    def gen_mixed_family_grouped_set(self, families=None, size=20000, count=1):
        """
        Generate grouped sets with multiple DGAs each, grouped by families.
        :param families: list of families to consider
        :param size: set size
        :param count: how many sets per family
        :return:
        """
        if not families:
            families = self.dga_data.families

        log.info('Generating mixed dga set for families: {!s}.'.format(families))

        for i in range(count):
            log.info('Generating set {!s}'.format(i))
            cur_set = self.grouped_random_set(size=size, families=families)
            cur_set.id = '{!s}_{!s}_{!s}_{!s}'.format(settings.SetTypes.mixed_dga_grouped_family.value, size, len(families), i)
            cur_set.serialize(where=settings.SETS_FOLDER + settings.SetTypes.mixed_dga_grouped_family.value + '/')
            self.data_sets_loaded.clear()

    def gen_single_grouped_day_set(self, families=None, size=20000, count=1, max_group_count=64):
        """
        Generate grouped sets with a single DGA, grouped by days. Only families considered that have at least 3 distinct days.
        :param families: list of families to consider
        :param size: set size
        :param count: how many sets per family
        :return:
        """
        if not families:
            families = self.dga_data.families

        log.info('Gernerating grouped daily sets for families: {!s}.'.format(families))

        for fam in families:
            for i in range(count):
                try:
                    cur_set = self.grouped_daily_random_set(fam, size=size, max_group_count=max_group_count)
                    cur_set.id = '{!s}_{!s}_{!s}_{!s}'.format(settings.SetTypes.single_dga_grouped_day.value, size, fam, i)
                    cur_set.serialize(where=settings.SETS_FOLDER + settings.SetTypes.single_dga_grouped_day.value + '/')
                    self.data_sets_loaded.clear()

                except ValueError as e:
                    log.info(str(e))


class DGArchiveJSON:
    """
    Class to represent a DGArchive JSON file in python.
    """
    def __init__(self, file: str):
        """

        :param file: path to file
        """
        self.file = file
        with open(file, encoding='utf-8') as f:
            self.data = json.load(f)

    def get_all_by_family(self, family_name: str, sample_size=None, return_as_seed_map=False):
        """
        This method yields cleaned domains
        :param family_name: name of the DGA family to get all
        :param sample_size:
        :return: list of all domains in the JSON file containing to the queried family
        """
        dga_domains = set()
        seed_map = dict()

        for k in self.data.keys():
            if family_name.strip() in k:
                if return_as_seed_map:
                    seed_map[k] = set(clean_domain_list(self.data[k], dga=True))
                else:
                    dga_domains.update(set(clean_domain_list(self.data[k], dga=True)))

        if not dga_domains and not seed_map:
            raise ValueError('No matching domains found for family name {}'.format(family_name))

        if sample_size:
            dga_domains = random.sample(dga_domains, k=sample_size)

        if return_as_seed_map:
            return seed_map
        else:
            return dga_domains


class DGArchive:
    """
    Class to represent all present DGArchive JSON files in a single object.
    """
    def __init__(self, days=0, unique_threshold=250, no_import=False):
        """

        :param days: # of days to import, 0 means all remaining
        """
        self.all_files = []
        self.missing_files = []
        self.current_json = None
        self.families = set()
        self.unique_domain_per_dga = defaultdict(set)
        self.unique_under_threshold = defaultdict(set)
        self.seed_map_cache = (None, None)
        self.daily_map_cache = (None, None)
        self.unique_threshold = unique_threshold

        log.info('Initializing DGA Archive data...')

        self.all_files = []
        for (_, _, files) in walk(DGARCHIVE_FODLER):
            self.all_files.extend(files)
            break

        self.all_files = [DGARCHIVE_FODLER + '/' + f for f in self.all_files if len(f) == len('2015_05_26')]
        self.missing_files = self.all_files
        if not no_import:
            self.import_n_days(days)

    def lazy_load_family_seed_map(self, family):
        """
        Lazy loads all unique domains for one DGA for all available DGArchive data and groups it by seeds.
        :param family: family to consider
        :return: dict containing the mapping seed -> domain set
        """
        # this method yields cleaned domain lists
        if self.seed_map_cache:
            if self.seed_map_cache[0] == family:
                log.info('Getting seed map from cache for {!s}'.format(family))
                return self.seed_map_cache[1]

        seed_map = defaultdict(set)
        for f in self.all_files:
            self.current_json = DGArchiveJSON(f)
            try:
                tmp_seed_map = self.current_json.get_all_by_family(family_name=family, return_as_seed_map=True)
                log.verbose(tmp_seed_map)
                for k in tmp_seed_map.keys():
                    seed_map[k].update(tmp_seed_map[k])
            except ValueError:
                log.debug('{!s} not containd in file {!s}. Skipping'.format(family, f))

        self.seed_map_cache = (family, seed_map)
        return seed_map

    def lazy_load_daily_seed_map(self, family):
        """
        Lazy load a map for one family. Key is the data, values are the corresponding DGA domains
        :param family:
        :return:
        """
        # this method yields cleaned domain lists

        if self.daily_map_cache:
            if self.daily_map_cache[0] == family:
                log.info('Getting day map from cache for {!s}'.format(family))
                return self.daily_map_cache[1]

        day_map = defaultdict(set)
        for f in self.all_files:
            self.current_json = DGArchiveJSON(f)
            try:
                day_map[os.path.basename(f)] = self.current_json.get_all_by_family(family_name=family)
            except ValueError:
                log.debug('{!s} not containd in file {!s}. Skipping'.format(family, f))

        self.daily_map_cache = (family, day_map)

        return day_map

    def import_n_days(self, days=0):
        """
        Import n more days from DGArchive
        :param days: # of days to import, 0 days mean all unimported days are imported
        :return:
        """
        keys_to_remove = []
        for fam in self.unique_under_threshold.keys():
            self.unique_domain_per_dga[fam] = self.unique_under_threshold[fam]
            keys_to_remove.append(fam)

        for k in keys_to_remove:
            self.unique_under_threshold.pop(k)

        if days != 0:
            files = random.sample(self.missing_files, days)
            self.missing_files = [f for f in self.missing_files if f not in files]
        else:
            files = self.missing_files

        log.info('Importing {!s} more days...'.format(len(files)))
        for s in files:
            self.current_json = DGArchiveJSON(s)
            self._collect_families()
            self._collect_unique_domains()

        keys_to_remove = []
        for fam in self.unique_domain_per_dga.keys():
            if len(self.unique_domain_per_dga[fam]) < self.unique_threshold:
                self.unique_under_threshold[fam] = self.unique_domain_per_dga[fam]
                keys_to_remove.append(fam)
        for k in keys_to_remove:
            self.unique_domain_per_dga.pop(k)

        self.families = set(self.unique_domain_per_dga.keys())

    def _collect_families(self):
        """
        Determines present families in the current JSON file.
        :return:
        """
        for k in self.current_json.data.keys():
            self.families.add(k.split('_')[0].strip())

    def _collect_unique_domains(self):
        """
        collects all unique domains present for each family
        :return:
        """
        for fam in self.families:
            try:
                self.unique_domain_per_dga[fam].update(self.current_json.get_all_by_family(fam))
            except ValueError:
                log.debug('The file {!s} does not contain family {!s}.'.format(self.current_json.file, fam))

    def random_sample(self, size, families=None, uniform=True, fill=False):
        """
        Yields random subsets according to given parameters.
        :param size: 
        :param families: 
        :param uniform:
        :return: 
        """
        # This method uses cleaned domains
        sample = set()
        if families:
            if not uniform:
                for f in families:
                    try:
                        sample.update(self.unique_domain_per_dga[f])
                    except Exception:
                        log.warning('Skipping family {!s}. The family does not exist.'.format(f))

                sample = random.sample(sample, size)
            else:
                sample_size = math.floor(size / len(families))
                for f in families:
                    if sample_size > len(self.unique_domain_per_dga[f]):
                        sample.update(self.unique_domain_per_dga[f])
                    else:
                        sample.update(random.sample(self.unique_domain_per_dga[f], sample_size))
                if size > len(sample) and fill:
                    sample.update(random.sample(self.__expand_unique_domains(), size - len(sample)))

        else:
            sample = random.sample(self.__expand_unique_domains(), size)

        return {LabeledData(d, 1) for d in sample}

    def __expand_unique_domains(self):
        """
        Yields all unique DGA domains.
        :return:
        """
        res = set()
        for k in self.unique_domain_per_dga.keys():
            res.update(self.unique_domain_per_dga[k])
        return res

    def extend_dga_data(self, days=0):
        self.import_n_days(days)

    def stats(self):
        log.info('Statistics for the current DGArchive set:\n '
                 'Family count: {!s}'.format(len(self.families)) +
                 '\nFamilies: {!s}'.format(self.families))
        log.info('Unique domain counts:')
        for f in self.families:
            log.info('{!s}: {!s}'.format(f, len(self.unique_domain_per_dga[f])))

"""
    def to_table(self):
        s = sorted(self.families)
        for f in s:
            print('{!s} & {!s} & {!s} & \\dn{!s} \\\\'.format(f.capitalize(), DGA_TYPE_MAP[f].upper(), '{0:,}'.format(len(self.unique_domain_per_dga[f])), '{' + random.sample(self.unique_domain_per_dga[f], 1)[0] + '}'))
"""


class BenignData:
    """
    Class to represent pcap files in python.
    """
    def __init__(self, consider_tuples=False, consider_nonnx=True, filter_non_valid_tld=True):
        self.tsharkcmd_nx = 'tshark -r {!s} -T fields -e dns.qry.name -e dns.qry.type -R "dns.flags.response eq 1 and dns.flags.rcode eq 3" -2'
        self.tsharkcmd_nonnx = 'tshark -r {!s} -T fields -e dns.qry.name -e dns.qry.type -R "dns.flags.response eq 1 and dns.flags.rcode eq 0" -2'
        self.unique_domains_file_nx = WORKSPACE_ROOT + 'unique_domains_cache.txt'
        self.unique_tuples_file = WORKSPACE_ROOT + 'unique_tuples_cache.txt'
        self.unique_non_nx_file = WORKSPACE_ROOT + 'unique_nonnx_domains_cache.txt'
        self.unique_domains_overall_nx = set()
        self.unique_domains_overall_nonnx = set()
        self.unique_tuples_overall = set()
        log.info('Initializing unique domain data from pcaps...')

        if path.isfile(self.unique_domains_file_nx):
            with open(self.unique_domains_file_nx, 'r', encoding='utf-8') as f:
                self._read_cache_file_line_by_line(f, self.unique_domains_overall_nx)

        if consider_nonnx:
            if path.isfile(self.unique_non_nx_file):
                with open(self.unique_non_nx_file, 'r', encoding='utf-8') as f:
                    self._read_cache_file_line_by_line(f, self.unique_domains_overall_nonnx)

        if consider_tuples:
            log.info('Initializing unique tuples data from pcaps...')
            if path.isfile(self.unique_tuples_file):
                with open(self.unique_tuples_file, 'r', encoding='utf-8') as f:
                    self._read_cache_file_line_by_line(f, self.unique_tuples_overall)
        else:
            log.info('Skipping tuples...')

    def _read_cache_file_line_by_line(self, f, store_in_set):
        """
        Read a cached domain file line by line. Inserts each line as one sample into the store_in_set set.
        :param f:
        :param store_in_set:
        :return:
        """
        while True:
            line = f.readline()
            if not line:
                break
            store_in_set.add(line.strip().lower())

    def clean_from_known_dga_domains(self, clean_from_guesses=False):
        """
        Clean loaded domains from known DGA domains using DGArchive.
        This may take a long time, as each domain has to be checked via the DGArchive API.
        :param clean_from_guesses:
        :return:
        """
        gu, hi = dgarchive_api.analyze_dga_qry_file()
        hits = set()
        guesses = set()
        for h in hi:
            hits.add(h['domain'])
        if clean_from_guesses:
            for g in gu:
                guesses.add(g['domain'])

        log.info('Starting filtering of known DGA domains. Size: {!s}'.format(len(self.unique_domains_overall_nx)))

        to_remove = set()

        for d in self.unique_domains_overall_nx:
            if d in hits:
                to_remove.add(d)
                log.debug('Removing known DGA domain: {!s}'.format(d))
        if clean_from_guesses:
            for d in self.unique_domains_overall_nx:
                if d in guesses:
                    to_remove.add(d)
                    log.debug('Removing guessed DGA domain: {!s}'.format(d))

        self.unique_domains_overall_nx.difference_update(to_remove)
        write_unique_set_to_file(self.unique_domains_overall_nx, self.unique_domains_file_nx)
        log.info('Cleaning complete. Size: {!s}. Removed: {!s}'.format(len(self.unique_domains_overall_nx), len(to_remove)))

    def stats(self):
        log.info('Statistics:\n Unique domain count: {!s} \n Unique tuples count: {!s}'.format(len(self.unique_domains_overall_nx), len(self.unique_tuples_overall)))
        log.info('Counts per RR Type:')
        counter = defaultdict(int)
        for t in self.unique_tuples_overall:
            domain, rr = t.split('\t')
            counter[rr] += 1
        for k in counter.keys():
            log.info('{!s}:\t{!s}'.format(k, counter[k]))

    def random_sample(self, k, nx=True, rr_types=None):
        """
        Get random subset of all domains in the workspace.
        If rr types is set it is filtered for these rr types before.
        :param k: 
        :param rr_types: a list of strings containing the rr types as in their digit representation according to the RFC
        :return: random sub sample
        """
        if nx:
            if rr_types is None:
                sample = random.sample(self.unique_domains_overall_nx, k)
                return {LabeledData(s, False) for s in sample}
            else:
                filtered = {d.split('\t')[0] for d in self.unique_tuples_overall if d.split('\t')[1] in rr_types}
                sample = random.sample(filtered, k)
                return {LabeledData(s, False) for s in sample}

        else:
            sample = random.sample(self.unique_domains_overall_nonnx, k)
            return {LabeledData(s, False) for s in sample}

    def import_siemens_data(self, folder: str):
        # 1505312701.958720000    1       up4l3v6febkebcvj5yq3lfuuhm.d58c983423e9b83e98a53256cf894caf.skey.ers.trendmicro.com
        log.info('Starting domain extraction. This may take some time...')

        all_csv = []
        for (_, _, files) in walk(folder):
            all_csv.extend(files)
            break

        all_csv = [folder + '/' + csv for csv in all_csv]

        for file in all_csv:

            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            lines = set(clean_domain_list({d.split(',')[1][1:-1].strip().lower() for d in lines if len(d.split(',')) > 2}))
            lines = filter_non_valid(lines)

            self.unique_domains_overall_nx.update(lines)

            del lines

        write_unique_set_to_file(self.unique_domains_overall_nx, self.unique_domains_file_nx)

    def import_from_pcap(self, folder: str, rr_types=None, copy=False, nx=True):
        """
        Import all pcaps from the given folder to the current workspace.
        :param folder: 
        :param rr_types: 
        :param copy: 
        :return: 
        """
        pcap_files = []
        tmp_pcaps = []

        for (_, _, files) in walk(folder):
            tmp_pcaps.extend(files)
            break

        tmp_pcaps = [folder + '/' + pcap for pcap in tmp_pcaps]

        if copy:
            log.info('Copying {!s} pcaps to local workspace. This may take a while.'.format(len(tmp_pcaps)))

            for p in tmp_pcaps:
                log.info('Copying {!s}'.format(p))
                pcap_files.append(copy2(p, PCAPS_FOLDER))
        else:
            pcap_files = tmp_pcaps

        log.info('Starting domain extraction. This may take some time...')

        for pcap in pcap_files:
            self.handle_pcap(pcap, nx=nx)


        self.unique_domains_overall_nx = set(clean_domain_list(self.unique_domains_overall_nx))
        self.unique_domains_overall_nonnx = set(clean_domain_list(self.unique_domains_overall_nonnx))

        self.unique_domains_overall_nx = filter_non_valid(self.unique_domains_overall_nx)

        if nx:
            write_unique_set_to_file(self.unique_domains_overall_nx, self.unique_domains_file_nx)
            write_unique_set_to_file(self.unique_tuples_overall, self.unique_tuples_file)
        else:
            write_unique_set_to_file(self.unique_domains_overall_nonnx, self.unique_non_nx_file)
        log.info('Domain extraction finished.')

    def handle_pcap(self, pcap, nx=True):
        log.info('Extraction started for: {!s}'.format(pcap))
        # self.pcaps.append(pyshark.FileCapture(pcap, keep_packets=False))
        unique_cur_pcap = set()

        if nx:
            tsharkcmd = self.tsharkcmd_nx
            to_add_to = self.unique_domains_overall_nx
            to_add_to_tuple = self.unique_tuples_overall
        else:
            tsharkcmd = self.tsharkcmd_nonnx
            to_add_to = self.unique_domains_overall_nonnx
            to_add_to_tuple = self.unique_tuples_overall

        process = subprocess.Popen(tsharkcmd.format(pcap), shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT)
        while True:
            domain = None
            try:
                nextline = process.stdout.readline()
                if process.poll() is not None:
                    break
                if nextline:
                    domain, rr_type = nextline.decode().lower().split('\t')
            except UnicodeDecodeError:
                log.debug('Decoding error.')
            except ValueError:
                log.debug('Value Error: {!s}'.format(nextline.decode()))
            if domain:
                domain_rr = '{!s}\t{!s}'.format(domain.strip(), rr_type.strip())

                to_add_to.add(domain.strip())
                to_add_to_tuple.add(domain_rr)
                unique_cur_pcap.add(domain_rr)
        if nx:
            write_unique_set_to_file(unique_cur_pcap, pcap + '_cache_uniques.txt')
        else:
            write_unique_set_to_file(unique_cur_pcap, pcap + '_cache_uniques_nonnx.txt')

        if process.returncode != 0:
            log.warning(
                'tshark exited with non-zero return code. Either no valid pcap or the pcap file is "cut-off" in the middle:\n {!s}'.format(
                    process.stderr))


class PublicSuffixes:
    """
    Represents the official public suffixes list maintained by Mozilla  https://publicsuffix.org/list/
    """
    def __init__(self, file=SCRIPTS_ROOT + '/bin/public_suffix.txt'):
        with open(file, encoding='utf-8') as f:
            self.data = f.readlines()

        self.data = clean_domain_list(self.data)
        self.data = ['.' + s for s in self.data if not (s.startswith('/') or s.startswith('*'))]
        self.data = clean_domain_list(self.data)

    def get_valid_tlds(self):
        return [s for s in self.data if len(s.split('.')) == 2]

    def get_valid_public_suffixes(self):
        return self.data


def write_unique_set_to_file(uniques, file):
    with open(file, 'w+', encoding='utf-8') as f:
        for l in uniques:
            f.write(l + '\n')


class AlexaTopX:

    def __init__(self, file=SCRIPTS_ROOT + '/bin/top-1m_alexa.csv', filter_pub_suf=True, x=1000000):
        with open(file, encoding='utf-8') as f:
            self.data = f.readlines()

        self.data = [d.split(',')[1] for d in self.data]

        self.data = self.data[0:x-1]

        pub_suf = PublicSuffixes().get_valid_public_suffixes()

        if filter_pub_suf:
            self.data = [d for d in self.data if d not in pub_suf]

        self.data = clean_domain_list(self.data)
