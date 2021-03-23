#!/usr/bin/python
# -*- coding: utf-8 -*-
# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2001, 2019. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************

'''
Created on Jul 18, 2019

@author: DAVIDMARTINEZIRAOLA
'''

import random

if __name__ == '__main__':
    with open("/home/jibmaird/Data/Corpora/ADE-v2/Exper/all.txt", "rb") as f:
        data = f.read().split('\n')
    test_size = len(data)*2/10

    with open("/home/jibmaird/Data/Corpora/ADE-v2/Exper/checked.txt", "rb") as f:
        checked = f.read().split('\n')

    random.shuffle(checked)

    test = checked[:test_size]
    print(test)
