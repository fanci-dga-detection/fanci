#!/usr/bin/env python3

import socket
import random
import string
from time import sleep


def random_domain(length, tld):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length)) + '.' + tld


if __name__ == '__main__':
    for i in range(250):
        domain = random_domain(12, 'de').strip()
        print(domain)
        try:
            socket.getaddrinfo(domain, 'www')
        except Exception as e:
            pass
        sleep(0.05)
