import os
os.chdir(os.path.expanduser("~/software/"))
import settings
from data_processing import data
from learning import classifiers
from learning import stats_metrics
from learning import eval_train_test
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC
from util import util
import logging
import numpy

settings.configure_logger()

w = data.Workspace(days=1, empty=True)
w.load_all()