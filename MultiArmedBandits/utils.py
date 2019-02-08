import numpy as np
import math


def secondLargest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None

def maxExcept(numbers, forbidden_id):
    a = max(numbers)
    a_id = numbers.index(a)
    if a_id == forbidden_id:
        a = secondLargest(numbers)
    return a

def smoothen(a_list):
    a = []
    for i in range(len(a_list) - 1):
        a.append((a_list[i] + a_list[i+1])/2)
    a.append(a_list[-1])
    return a