import logging
import re
from collections import defaultdict, Counter

import numpy
from scipy import stats
from sklearn.externals.joblib.parallel import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer

from data_processing.data import PublicSuffixes
from learning.exceptions import UnsuitableFeatureOrderException

log = logging.getLogger('log')


HEX_DIGITS = set('0123456789abcdef')
VOWELS = set('aeiou')
PARTS_MAX_CONSIDERED = 4

PUBLIC_SUF = PublicSuffixes()
VALID_TLDS = PUBLIC_SUF.get_valid_tlds()  # TLD source https://publicsuffix.org/
VALID_PUB_SUFFIXES = PUBLIC_SUF.get_valid_public_suffixes()

__domain = ''
__dot_split = ()
__joined_dot_split = ''
__dot_split_suffix_free = ()
__joined_dot_split_suffix_free = ''
__public_suffix = ''
__unigram = ()


def extract_features(d: str, features):
    """
    Extract all features given as arguments from the given domain
    :param features: arbitrary many features, names of the public functions
    :param d: domain name as string
    :param debug: set to true, to return a tuple of a scaled and unscaled feature vector
    :return: scaled feature vector according to input data (in case debug is set to True: feature_vector, scaled_vector)
    """
    feature_vector = []

    __fill_cache(d)

    log.verbose('Extracting features for: {!s}'.format(d))
    log.verbose('Cache: {!s}\n{!s}\n{!s}\n{!s}\n{!s}'.format(__dot_split, __joined_dot_split, __dot_split_suffix_free,
                                                           __joined_dot_split_suffix_free, __public_suffix))

    # workaround for defect domain data in the benign set (actually should have been filtered out, but one always forget cases)
    # sadly not working, kept here for later analysis maybe
    """
    if len(__joined_dot_split_suffix_free) == 0:
        log.warning('Defect domain with non-TLD length of zero: {!s}'.format(d))
        return [0] * len(features)
    """
    # using exception here for more robustness due to defect data + performance better than if else statements
    for f in features:
        try:
            feature_vector = feature_vector + f()
        except (ValueError, ArithmeticError) as e:
            # XXX maybe better approach than setting to zero?
            log.error('Feature {!s} could not be extracted of {!s}. Setting feature to zero'.format(f, d))
            feature_vector = feature_vector + [0]

    log.verbose('\n{!s}, {!s}, {!s}'.format(d, [f.__name__ for f in features], feature_vector))

    return feature_vector


def __fill_cache(domain: str):
    global __dot_split, __joined_dot_split, __dot_split_suffix_free, __joined_dot_split_suffix_free, __public_suffix, __domain
    __domain = domain
    __dot_split = tuple(domain.split('.'))
    __joined_dot_split = ''.join(list(__dot_split))
    __dot_split_suffix_free, __public_suffix = __public_suffix_remover(__dot_split)
    __joined_dot_split_suffix_free = ''.join(__dot_split_suffix_free)


def __public_suffix_remover(dot_split):
    """
    Finds the largest matching public suffix
    :param dot_split: 
    :return: public suffix free domain as dot split, public suffix
    """
    match = ''

    if len(dot_split) < 2:
        return tuple(dot_split), match

    for i in range(0, len(dot_split)):
        sliced_domain_parts = dot_split[i:]
        match = '.' + '.'.join(sliced_domain_parts)
        if match in VALID_PUB_SUFFIXES:
            cleared = dot_split[0:i]
            return tuple(cleared), match
    return tuple(dot_split), match


def _vowel_ratio():
    """
    Ratio of vowels to non-vowels
    :return: vowel ratio
    """
    vowel_count = 0
    alpha_count = 0
    domain = __joined_dot_split_suffix_free
    for c in domain:
        if c in VOWELS:
            vowel_count += 1
        if c.isalpha():
            alpha_count += 1

    if alpha_count > 0:
        return [vowel_count/alpha_count]
    else:
        return [0]


def _digit_ratio():
    """
    Determine ratio of digits to domain length
    :return:
    """
    domain = __joined_dot_split_suffix_free
    digit_count = 0
    for c in domain:
        if c.isdigit():
            digit_count += 1

    return [digit_count/len(domain)]


def _length():
    """
    Determine domain length
    :return:
    """
    return [len(__domain)]


def _contains_wwwdot():
    """
    1 if 'www. is contained' 0 else
    :return:
    """
    if 'www.' in __domain:
        return [1]
    else:
        return [0]


def _contains_subdomain_of_only_digits():
    """
    Checks if subdomains of only digits are contained.
    :return: 
    """
    for p in __dot_split:
        only_digits = True
        for c in p:
            if c.isalpha():
                only_digits = False
                break
        if only_digits:
            return [1]
    return [0]


def _subdomain_lengths_mean():
    """
    Calculates average subdomain length
    :return:
    """
    overall_len = 0
    for p in __dot_split_suffix_free:
        overall_len += len(p)
    return [overall_len / len(__dot_split_suffix_free)]


def _parts():
    """
    Calculate the number of domain levels present in a domain, where rwth-aachen.de evaluates to 1 -> [1,0,0,0,0]
    The feature is decoded in a binary categorical way in the form [0,0,0,1,0]. The index represents the number of subdomains
    :return:
    """

    feature = [0] * PARTS_MAX_CONSIDERED
    split_length = len(__dot_split_suffix_free)
    if split_length >= PARTS_MAX_CONSIDERED:
        feature[PARTS_MAX_CONSIDERED - 1] = 1
    else:
        feature[split_length - 1] = 1

    return feature


def _contains_ipv4_addr():
    """
    check if the domain contains a valid IP address. Considers both, v4 and v6
    :return:
    """
    match_v4 = re.search(ipv4_pattern, __domain)
    if match_v4:
        return [1]
    else:
        return [0]


def _contains_digits():
    if any(char.isdigit() for char in __domain):
        return [1]
    else:
        return [0]


def _has_valid_tld():
    """
    Checks if the domain ends with a valid TLD
    :return:
    """
    if __public_suffix:
        return [1]
    return [0]


def _contains_one_char_subdomains():
    """
    Checks if the domain contains subdomains of only one character
    :return:
    """
    parts = __dot_split

    if len(parts) > 2:
        parts = parts[:-1]

    for p in parts:
        if len(p) == 1:
            return [1]

    return [0]


def _prefix_repetition():
    """
    Checks if the string is prefix repeating exclusively.
    Example: 123123 and abcabcabc are prefix repeating 1231234 and ababde are not.
    :return: 
    """
    i = (__domain + __domain).find(__domain, 1, -1)
    return [0] if i == -1 else [1]


def _char_diversity():
    """
    counts different characters, divided by domain length. 
    :return: 
    """
    counter = defaultdict(int)

    domain = __joined_dot_split_suffix_free
    for c in domain:
        counter[c] += 1

    return [len(counter)/len(domain)]


def _contains_tld_as_infix():
    """
    Checks for infixes that are valid TLD endings like .de in 123.de.rwth-aachen.de
    If such a infix is found 1 is returned, 0 else
    :return:
    """
    for tld in VALID_TLDS:
        if tld[1:] in __dot_split_suffix_free:
            return [1]
    return [0]


def _n_grams():
    """
    Calculates various statistical features over the 1-,2- and 3-grams of the suffix and dot free domain
    :return: 
    """
    global __unigram
    feature = []

    for i in range(1,4):
        ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(i, i))
        counts = ngram_vectorizer.build_analyzer()(__joined_dot_split_suffix_free)
        npa = numpy.array(list(Counter(counts).values()), dtype=int)
        if i == 1:
            __unigram = npa

        feature += __stats_over_n_grams(npa)

    return feature


def __stats_over_n_grams(npa):
    """
    Calculates statistical features over ngrams decoded in numpy arrays
    stddev, median, mean, min, max, quartils, alphabetsize (length of the ngram)
    :param npa: 
    :return: 
    """
    # TODO check for Hoover Index, Gini Coef, Rosenbluth-Index, Wölbung and Schiefe

    if npa.size > 0:
        stats = [npa.std(), numpy.median(npa), npa.mean(), numpy.min(npa), numpy.max(npa), numpy.percentile(npa, 25),
             numpy.percentile(npa, 75)]
    else:
        stats = [-1, -1, -1, -1, -1, -1, -1]

    return stats


def _alphabet_size():
    """
    Calculates the alphabet size of the domain
    :return: 
    """
    if __unigram is ():
        raise UnsuitableFeatureOrderException('The feature _n_grams has to be calculated before.')
    return [len(__unigram)]


def _shannon_entropy():
    """
    Calculates the Shannon Entropy based on the frequencies of characters.
    :return: 
    """
    # Note for me: checked against an online calculator for verification: http://www.shannonentropy.netmark.pl/
    if __unigram is ():
        raise UnsuitableFeatureOrderException('The feature _n_grams has to be calculated before.')

    return [stats.entropy(__unigram, base=2)]


def _hex_part_ratio():
    """
    Counts all parts that are only hex. Normalized by the overall part count
    :return: 
    """
    hex_parts = 0
    for p in __dot_split_suffix_free:
        if all(c in HEX_DIGITS for c in p):
            hex_parts += 1

    return[hex_parts / len(__dot_split_suffix_free)]


def _underscore_ratio():
    """
    Calculates the ratio of occuring underscores in all domain parts excluding the public suffix
    :return: 
    """
    underscore_counter = 0
    for c in __joined_dot_split_suffix_free:
        if c == '_':
            underscore_counter += 1

    return [underscore_counter / len(__joined_dot_split_suffix_free)]


def _ratio_of_repeated_chars():
    """
    Calculates the ratio of characters repeating in the string
    :return: 
    """
    # TODO maybe weighted? check the impact
    if __unigram is ():
        raise UnsuitableFeatureOrderException('The feature _n_grams has to be calculated before.')

    repeating = 0
    for i in __unigram:
        if i > 1:
            repeating += 1
    return [repeating / len(__unigram)]


def _consecutive_consonant_ratio():
    """
    Calculates the ratio of conescutive consonants
    :return: 
    """
    # TODO weighted: long sequences -> higher weight

    consecutive_counter = 0
    for p in __dot_split_suffix_free:
        counter = 0
        i = 0
        for c in p:
            if c.isalpha() and c not in VOWELS:
                counter +=1
            else:
                if counter > 1:
                    consecutive_counter += counter
                counter = 0
            i += 1
            if i == len(p) and counter > 1:
                consecutive_counter += counter

    return [consecutive_counter / len(__joined_dot_split_suffix_free)]


def _consecutive_digits_ratio():
    """
    Calculates the ratio of consecutive digits
    :return: 
    """

    consecutive_counter = 0
    for p in __dot_split_suffix_free:
        counter = 0
        i = 0
        for c in p:
            if c.isdigit():
                counter +=1
            else:
                if counter > 1:
                    consecutive_counter += counter
                counter = 0
            i += 1
            if i == len(p) and counter > 1:
                consecutive_counter += counter

    return [consecutive_counter / len(__joined_dot_split_suffix_free)]


# Pattern matching ipv4 addresses according to RFC
ipv4_pattern = re.compile("(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])")
ALL_FEATURES = _length, _parts, _vowel_ratio, _digit_ratio, _contains_ipv4_addr, _contains_digits, \
                        _has_valid_tld, _contains_one_char_subdomains, _contains_wwwdot, _subdomain_lengths_mean, \
                        _prefix_repetition, _char_diversity, _contains_subdomain_of_only_digits, _contains_tld_as_infix, \
                        _n_grams, _hex_part_ratio, _underscore_ratio, _alphabet_size, _shannon_entropy, \
                        _ratio_of_repeated_chars, _consecutive_consonant_ratio, _consecutive_digits_ratio


def extract_all_features(data, n_jobs=-1):
    """
    Function extracting all available features to a numpy feature array.
    :param data: iterable containing domain name strings
    :return: feature matrix as numpy array
    """
    parallel = Parallel(n_jobs=n_jobs, verbose=1)
    feature_matrix = parallel(
        delayed(extract_features)(d, ALL_FEATURES)
        for d in data
        )

    return numpy.array(feature_matrix)


def extract_all_features_single(d):
    """
    Extracts all features of a single domain name
    :param d: string, domain name
    :return: extracted features as numpy array
    """
    return numpy.array(extract_features(d, ALL_FEATURES))
