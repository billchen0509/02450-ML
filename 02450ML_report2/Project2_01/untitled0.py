# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:16:40 2019

@author: Lychen
"""
import time


M = [['1', '2', '3', '4'], ['5', '6', '7', '8'], ['9', '10', '11', '12']]

start = time.time()
'''
def transpose2(matrix):
    row = len(matrix)
    column = len(matrix[0])
    if column > row:
        for i in range(column-row):
            matrix.append([])
            for j in range(column):
                matrix[row+i].append('0')
    elif column < row:
        for i in range(row):
            for j in range(row-column):
                matrix[i].append('0')

    d = max(row, column)
    for i in range(d):
        for j in range(d):
            if i < j:
                a = matrix[i][j]
                matrix[i][j] = matrix[j][i]
                matrix[j][i] = a
            else:
                pass

    if column > row:
        for i in range(column):
            for j in range(column-row):
                matrix[i].pop()
    if column < row:
        for i in range(row-column):
            matrix.pop()


transpose2(M)
'''
import numpy as np
np.asarray(M).T
end = time.time()
print('Start: {}, End: {}, Time: {}'.format(start, end, end-start))