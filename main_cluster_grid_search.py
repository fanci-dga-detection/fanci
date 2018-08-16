import argparse
import sys

sys.path.append('./')

import main_helper


def parse_args():
    """
    Parse the command line options.
    :return: parsed args obj
    """
    parser = argparse.ArgumentParser('Perform grid searched with different sets. Optimized to utilize array jobs on the HPC Cluster.')
    parser.add_argument('set', help='data set to choose', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.set == 1:
        main_helper.start_x_trained_y_test(n_jobs=-1, clf_type='rf')
    elif args.set == 2:
        main_helper.start_x_trained_y_test(n_jobs=-1, clf_type='svm')
    elif args.set == 3:
        main_helper.start_x_trained_y_test(n_jobs=-1, clf_type='rf', x='rwth')
    elif args.set == 4:
        main_helper.start_x_trained_y_test(n_jobs=-1, clf_type='svm', x='rwth')

    elif args.set == 5:
        main_helper.start_mix_dga_kfold(clf_type='svm')
    elif args.set == 6:
        main_helper.start_mix_dga_kfold(clf_type='rf')
    sys.exit(0) # necessary for cluster case, else seem not to exit gracefully
