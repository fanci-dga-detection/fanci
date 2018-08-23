#!/usr/bin/env python3

import argparse

import datetime
import json

import logging

import pickle
import requests

import settings


no_succ_list = []
retry_cnt = 5
log = logging.getLogger('log')
DGA_RWTH_QRY_FILE = settings.ANALYSIS_FOLDER + 'siemens_dga_qry_res.pkl'


def parse_args():
    parser = argparse.ArgumentParser(description='Query domainlists from DGArchive.')
    parser.add_argument('-d', '--date', type=str, help='date to query domains for. Use the form YYYY/MM/DD')
    parser.add_argument('-f', '--future', help='queries all domains from now on up to 1 year in the future', action='store_true')
    parser.add_argument('output', type=str, help='path to output folder')
    return parser.parse_args()


def download_for_date(date):

    print('Downloading domains for date: {!s}'.format(date))

    login = ('abc', 'xxx') # XXX enter your login here
    response = requests.get('https://dgarchive.caad.fkie.fraunhofer.de/d/{}'.format(date), auth=login)

    print('HTTP response: {}'.format(response))

    if response.status_code != 200:
        no_succ_list.append(date)
        return False
    else:
        print('Downloading successful')

    with open(cli_args.output + '/' + date.replace('/', '_'), 'wb+') as f:
        f.write(response.content)

    return True


def download_all_future_from_now():
    download_all_future(datetime.datetime.now())


def download_all_future(start_date):
    global retry_cnt

    now = start_date
    i = 0
    while i < 10:
        now_string = now.strftime('%Y/%m/%d')
        success = download_for_date(now_string)
        if success:
            now = now + datetime.timedelta(days=1)
            i += 1
        else:
            if retry_cnt <= 0:
                now = now + datetime.timedelta(days=1)
                i += 1
                retry_cnt = 5

            print('Retrying...')
            retry_cnt -= 1


def check_for_dga_domains(domains, persist=True):
    exclude_suffixes = settings.LOCAL_FILTER_SET_RWTH + settings.LOCAL_FILTER_SET_SIEMENS
    classification = []

    max_len = 50

    cur_slice = []

    progress = 0

    for d in domains:
        d = d.strip()
        d = d.lower()
        exclude = False
        for suf in exclude_suffixes:
            if d.endswith(suf):
                exclude = True
            if '.' not in d:
                exclude = True

        if not exclude:
            cur_slice.append(d)

        if len(cur_slice) == 100:
            log.debug(str(cur_slice))
            progress += len(cur_slice)
            log.info('{!s}/{!s}'.format(progress, len(domains)))
            login = ('xxx', 'xxx') # TODO
            response = requests.post('https://dgarchive.caad.fkie.fraunhofer.de/reverse',('\n'.join(cur_slice)).encode('utf-8'), auth=login)
            if response.status_code != 200:
                log.warning('Non 200 status code returned')
            else:
                # log.info(response.content.decode('utf-8'))
                classification.append(response.content)
            cur_slice = []

        if len(classification) > max_len:
            if persist:
                pickle_classification(classification)
            max_len += 50

    if len(cur_slice) > 0:
        login = ('xxx', 'xxx') # TODO
        response = requests.post('https://dgarchive.caad.fkie.fraunhofer.de/reverse', ('\n'.join(cur_slice)).encode('utf-8'), auth=login)
        if response.status_code != 200:
            log.warning('Non 200 status code returned')
        else:
            # log.info(response.content.decode('utf-8'))
            classification.append(response.content)

    if persist:
        pickle_classification(classification)

    return classification


def pickle_classification(classification):
    with open(DGA_RWTH_QRY_FILE, 'wb') as f:
        pickle.dump(classification, f)


def analyze_dga_qry_file(load=True, qry_list=None):
    # keys: 'guesses', 'hits', 'misses'

    if load:
        with open(DGA_RWTH_QRY_FILE, 'rb') as f:
            qry_list = pickle.load(f)
            return analyze_qry_list(qry_list)


def analyze_qry_list(qry_list):
    guesses = []
    hits = []
    for q in qry_list:
        q = q.decode('utf-8')
        as_json = json.loads(q)
        guesses.extend(as_json['guesses'])
        hits.extend(as_json['hits'])
    return guesses, hits


if __name__ == '__main__':
    cli_args = parse_args()
    if cli_args.date and cli_args.future:
        date = datetime.datetime.strptime(cli_args.date, '%Y/%m/%d')
        download_all_future(date)
    elif cli_args.future:
        download_all_future_from_now()
    elif cli_args.date:
        download_for_date(cli_args.date)
    else:
        print('Please provide valid arguments.')
