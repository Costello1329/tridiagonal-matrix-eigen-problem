import numpy

a = numpy.array([[1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [5, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 6, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 8, 5, 89, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 89, 6, 9, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 9, 7, 12, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 12, 8, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 2, 9, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 8, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 765, 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 34], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 3]])
w, v = numpy.linalg.eig(a)
print(w, v)

# [765.01054369 -84.32041186  95.31754558  19.66602371  -5.83018005
#   -4.5893017    1.84970367  -0.97572412   9.99813369   9.82822776
#    8.0506992  -31.00251297  36.99725339]
