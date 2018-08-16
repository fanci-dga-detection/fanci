import numpy
import numpy as np
import matplotlib
from sklearn.feature_extraction.text import CountVectorizer

from data_processing.data import LabeledData, BenignData
from learning import feature_extraction
from learning.feature_extraction import _char_diversity, extract_features, _contains_tld_as_infix, ipv4_pattern
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
import re


def repetitions(s):
    r = re.compile(r"(.+?)\1+")
    for match in r.finditer(s):
        yield (match.group(1), len(match.group(0))/len(match.group(1)))


from collections import defaultdict, Counter


def getsubs(loc, s):
    substr = s[loc:]
    i = -1
    while(substr):
        yield substr
        substr = s[loc:i]
        i -= 1


def _contains_ip_addr(__domain):
    """
    check if the domain contains a valid IP address. Considers both, v4 and v6
    :return:
    """
    match_v4 = re.search(ipv4_pattern, __domain)
    print(__domain)
    print(match_v4)
    if match_v4:
        return [1]
    else:
        return [0]

if __name__ == '__main__':
    res = []
    with open('/Users/samuel/Downloads/working.txt', 'r') as f:
        all_lines = f.readlines()
        i = 0
        while i < len(all_lines):
            res.append(all_lines[i].strip() + all_lines[i+1])
            i += 2


    with open('/Users/samuel/Downloads/out.txt', 'w') as f:
        f.writelines(res)
