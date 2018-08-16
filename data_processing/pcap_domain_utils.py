#!/usr/bin/env python3

import argparse
import json
import logging

import operator

import os
from os import walk

import subprocess

import settings
from data_processing import dgarchive_api

log = logging.getLogger('log')

NXDOMAIN_RETURN_CODE = 3
dec_err = []


def parse_args():
    parser = argparse.ArgumentParser(description='Reads a pcap file and filters all domain names if all NXDOMAIN responses insisde the pcap.')
    parser.add_argument('input', type=str, help='path to text file containing one domain per row')
    parser.add_argument('output', type=str, help='path to output directory')
    parser.add_argument('-sld', action='store_true', help='Set this to additionally output a histogram for the sld occurences.')

    return parser.parse_args()


histogram_out_file = ''
sld_out_file = ''


def mapback_dga_domains(pcap_folder=None):

    guess, hit = dgarchive_api.analyze_dga_qry_file()

    hit_result_list = []

    log.info('Starting back mapping of DGA domains.')

    pcaps = []
    for (path, dirs, files) in walk(pcap_folder):
        pcaps.extend(files)
        break

    pcaps = [pcap_folder + '/' + pcap for pcap in pcaps]

    for pcap in pcaps:
        log.info('Processing: {!s}'.format(pcap))
        expanded = pcap_expander(pcap)

        for d in expanded.keys():
            for h in hit:
                if h['domain'].strip() == d:
                    for metadata in expanded[d]:
                        hit_result_list.append(
                            (h['domain'], metadata[0], metadata[1], metadata[2], h['family'], h['validity']['from'], h['validity']['to']))

    log.info('Writing results to disk.')
    with open(settings.ANALYSIS_FOLDER + 'mapback_{!s}'.format(settings.NOW_STR), 'w', encoding='utf-8') as f:
        f.write('\t'.join(('domain', 'src', 'dst', 'when', 'DGA Family', 'DGA domain valid from', 'DGA domain valid to')) + '\n' + '\n')
        for l in hit_result_list:
            f.write('\t'.join(l) + '\n')
    log.info('Search finished.')


def mapback_dga_siemens(folder):
    log.info('Starting domain extraction. This may take some time...')

    all_csv = []
    for (_, _, files) in walk(folder):
        all_csv.extend(files)
        break

    all_csv = [folder + '/' + csv for csv in all_csv]

    log.info('Analyzing qry file')

    _, hit = dgarchive_api.analyze_dga_qry_file()

    dga_domains = set()

    for dga in hit:
        dga_domains.add(dga['domain'])

    res = dict()

    for file in all_csv:
        log.info('Processing: {!s}'.format(file))

        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        base = os.path.basename(file)

        lines = [(d.split(',')[1][1:-1].strip().lower(), d.split(',')[0][1:-1], d.split(',')[3][1:-1], base) for d in lines if len(d.split(',')) > 3]

        for l in lines:
            if l[0] in dga_domains:
                if l[0] not in res:
                    dga_info = None
                    for dga in hit:
                        if dga['domain'] == l[0]:
                            dga_info = dga
                            break
                    res[l[0]] = dga_info, [l]
                else:
                    res[l[0]][1].append(l)
        del lines

    with open(settings.WORKSPACE_ROOT + '/siemens_hits_resolved.json', 'w') as f:
        f.write(json.dumps(res, sort_keys=True, indent=4, ))

    return res


def pcap_iterator(pcap):
    tsharkcmd = 'tshark -r {!s} -T fields -e dns.qry.name -e ip.src -e ip.dst -e frame.time -R "dns.flags.response eq 1 and dns.flags.rcode eq 3" -2'

    process = subprocess.Popen(tsharkcmd.format(pcap), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        try:
            nextline = process.stdout.readline()
            if process.poll() is not None:
                break
            if nextline:
                domain, src, dst, when = nextline.decode().strip().split('\t')
                domain = domain.strip().lower()
                yield domain, src, dst, when

        except UnicodeDecodeError:
            log.debug('Decoding error.')
        except ValueError:
            log.debug('Value Error: {!s}'.format(nextline.decode()))

    if process.returncode != 0:
        log.warning(
            'tshark exited with non-zero return code. Either no valid pcap or the pcap file is "cut-off" in the middle:\n {!s}'.format(
                process.stderr))


def pcap_expander(pcap):
    """
    Expands a pcap containing DNS traffic to a dict of the form: domain -> [(source, destination, when)]
    :param path_to_pcaps_folder: path to folder containing pcap files
    :return:
    """

    res = dict()

    for domain, src, dst, when in pcap_iterator(pcap):
        if domain not in res:
           res[domain] = [(src, dst, when)]
        else:
           res[domain].append((src, dst, when))

    return res


# XXX ignore wildcard only request '*'
def histogram_unique(all_domains):
    log.info('Generating histogram for unique domains...')

    domain_counter = dict()

    for d in all_domains:
        if d not in domain_counter:
            domain_counter[d] = 1
        else:
            domain_counter[d] += 1

    return domain_counter


def collect_all_domains():
    with open(cli_args.input, 'r', errors='ignore') as f:
        all_domains = f.read().splitlines()
    return all_domains


def histogram_xld(all_domains, x=-2):
    log.info('Generating histogram for unique XLD domains...')

    domain_counter = dict()

    for d in all_domains:
        try:
            d = '.'.join(d.split('.')[x:])
            if d not in domain_counter:
                domain_counter[d] = 1
            else:
                domain_counter[d] += 1
        except ValueError:
            continue

    return domain_counter


def sort_domain_counter_dict(c_dict: dict):
    """
    Returns a sorted list of tuples of a counting dict
    :param c_dict: counter dict, mapping domain (str) -> count (int)
    :return: a sorted histogram of the domain counts, i.e. in the way list of tuples of the form (domain, count)
    """
    sorted_domains = sorted(c_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_domains


if __name__ == '__main__':
    cli_args = parse_args()
    log.info('Starting parsing of domains.')

    histogram_out_file = os.path.basename(cli_args.input) + '_histo_unique.txt'
    sld_out_file = os.path.basename(cli_args.input) + '_histo_sld_unique.txt'

    all_domains = collect_all_domains()


    domain_counter = histogram_unique(all_domains)
    s = sort_domain_counter_dict(domain_counter)
    with open(cli_args.output + '/' + histogram_out_file, 'a+') as f:
        for d in s:
            f.write(str(d[1]) + ' \t ' + str(d[0]) + '\n')

    if cli_args.sld:
        domain_counter = histogram_xld(all_domains)
        s = sort_domain_counter_dict(domain_counter)
        with open(cli_args.output + '/' + sld_out_file, 'a+') as f:
            for d in s:
                f.write(str(d[1]) + ' \t ' + str(d[0]) + '\n')
