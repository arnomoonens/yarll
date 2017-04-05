#!/usr/bin/env python
# -*- coding: utf8 -*-

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("experiment", type=str, help="JSON file with the experiment specification")

def main():
    args = parser.parse_args()

if __name__ == '__main__':
    main()
