import numpy as np
from numba import jit, prange
from time import time

t0 = time()
@jit(nopython=True, parallel=True)
def partition(array, low, high):
  pivot = array[high]
  i = low - 1
  for j in prange(low, high):
    if array[j] <= pivot:
      i = i + 1
      (array[i], array[j]) = (array[j], array[i])
  (array[i + 1], array[high]) = (array[high], array[i + 1])
  return i + 1


def quickSort(array, low, high):
  if low < high:
    pi = partition(array, low, high)
    quickSort(array, low, pi - 1)
    quickSort(array, pi + 1, high)

N = 10000000
data = np.zeros(N, dtype="int")
data = np.random.randint(low=0,high=N,size=N)
print("Unsorted Array")
print(data)

size = len(data)

quickSort(data, 0, size - 1)

print('Sorted Array in Ascending Order:')
print(data)
t1 = time()
print('openmp takes {}'.format(t1-t0))

#  NUMBA_NUM_THREADS=4 python openmp.py

