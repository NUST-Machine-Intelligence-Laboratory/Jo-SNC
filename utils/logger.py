# -*- coding: utf-8 -*-
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   @File        : logger.py.py
#   @Author      : Zeren Sun
#   @Created date: 2022/11/18 10:56
#   @Description :
#
# ================================================================
import os
from datetime import datetime
from easydict import EasyDict

# ---------- Color Printing ----------
PStyle = EasyDict({
    'end': '\33[0m',
    'bold': '\33[1m',
    'italic': '\33[3m',
    'underline': '\33[4m',
    'selected': '\33[7m',
    'red': '\33[31m',
    'green': '\33[32m',
    'yellow': '\33[33m',
    'blue': '\33[34m'
})


# ---------- Naive Print Tools ----------
def print_to_logfile(logfile, content, init=False, end='\n'):
    if init:
        with open(logfile, 'w') as f:
            f.write(content + end)
    else:
        with open(logfile, 'a') as f:
            f.write(content + end)


def print_to_console(content, style=None, color=None):
    flag = 0
    if color in PStyle.keys():
        content = f'{PStyle[color]}{content}'
        flag += 1
    if style in PStyle.keys():
        content = f'{PStyle[style]}{content}'
        flag += 1
    if flag > 0:
        content = f'{content}{PStyle.end}'
    print(content, flush=True)


# ---------- Simple Logger ----------
class Logger(object):
    def __init__(self, logging_dir, DEBUG=False, INFO_DISPLAY_TIMESTAMP=False):
        # set up logging directory
        self.DEBUG = DEBUG
        self.INFO_DISPLAY_TIMESTAMP = INFO_DISPLAY_TIMESTAMP
        self.logging_dir = logging_dir
        self.logfile_path = None
        self.debug_info_path = None
        self.msg_info_path = None
        os.makedirs(self.logging_dir, exist_ok=True)

    def set_logfile(self, logfile_name):
        f = open(f'{self.logging_dir}/{logfile_name}', 'w')
        f.close()
        f = open(f'{self.logging_dir}/debug-{logfile_name}', 'w')
        f.close()
        f = open(f'{self.logging_dir}/msg-{logfile_name}', 'w')
        self.logfile_path = f'{self.logging_dir}/{logfile_name}'
        self.debug_info_path = f'{self.logging_dir}/debug-{logfile_name}'
        self.msg_info_path = f'{self.logging_dir}/msg-{logfile_name}'

    def debug(self, content):
        if self.DEBUG:
            assert self.debug_info_path is not None
            print_to_logfile(logfile=self.debug_info_path, content=content, init=False)
        # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # print_to_console(f'{PStyle.green}{timestamp}{PStyle.end} - | {PStyle.yellow}DEBUG{PStyle.end}    | - {PStyle.yellow}{content}{PStyle.end}')

    def info(self, content):
        assert self.logfile_path is not None
        print_to_logfile(logfile=self.logfile_path, content=content, init=False)
        if self.INFO_DISPLAY_TIMESTAMP:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print_to_console(f'{PStyle.green}{timestamp}{PStyle.end} - | {PStyle.blue}INFO{PStyle.end}     | - {PStyle.blue}{content}{PStyle.end}')
        else:
            print_to_console(f'{PStyle.blue}{content}{PStyle.end}')

    def msg(self, content):
        print_to_console(f'{PStyle.yellow}{content}{PStyle.end}')
        assert self.msg_info_path is not None
        print_to_logfile(logfile=self.msg_info_path, content=content, init=False)


# ---------- Simple Writer ----------
class Writer(object):
    def __init__(self, root_dir, filename, header=None):
        self.root_dir = root_dir
        self.filename = filename
        assert os.path.isdir(self.root_dir), f'{root_dir} does not exist.'
        self.filepath = os.path.join(self.root_dir, self.filename)
        if header is not None:
            print_to_logfile(self.filepath, header, init=True)
        else:
            f = open(self.filepath, 'w')
            f.close()

    def write(self, content):
        print_to_logfile(self.filepath, content, init=False)
