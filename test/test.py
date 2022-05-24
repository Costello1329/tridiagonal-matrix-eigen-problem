from subprocess import PIPE, Popen, run
import matplotlib.pyplot as plt
from termcolor import cprint
import random
import numpy
import time
import os


def compile_code():
    process = Popen("clang++ -std=c++20 -o main -lpthread src/main.cpp", stdout=PIPE, stderr=PIPE, shell=True)

    _, errors = process.communicate()
    errors = errors.decode("utf-8")

    if len(errors) != 0:
        cprint("TEST CRASHED (Compiler error)", "red")
        cprint(errors, "red")
        exit(1)


def generate():
    tests = []
    for t in range(256):
        n = t + 1
        m = [random.uniform(1, 100) * random.choice([-1, 1]) for _ in range(2 * n - 1)]
        tests.append("{} {}\n".format(n, " ".join([str(el) for el in m])))
    return tests


def process_output (n, got):
    return [float(line.split(":")[0]) for line in got.split("\n")[-(n + 1):-1]]


def matrix_from_test(test):
    test = test.split(" ")
    elements = [float(el) for el in test[1:]]
    n = len(elements) // 2 + 1

    m = [[0. for j in range(n)] for i in range(n)]
    for i in range(n):
        m[i][i] = elements[i]
    for i in range(n - 1):
        m[i + 1][i] = elements[n + i]
        m[i][i + 1] = elements[n + i]
    return n, m


def solve(test):
    n, m = matrix_from_test(test)
    w, _ = numpy.linalg.eig(numpy.matrix(m))
    return n, [el for el in w]


def compare(first, second):
    return len(first) == len(second) and all([abs(el1 - el2) < 1e-3 for el1, el2 in zip(first, second)])


def test_all():
    compile_code()

    plot_n = []
    plot_t = []

    for test in generate():
        start = time.time()

        got, errors = [t.decode("utf-8") for t in
            Popen("./main", stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True).communicate(input=bytes(test, "ascii"))]

        end = time.time()

        if len(errors) != 0:
            cprint("TEST CRASHED (Runtime error)", "red")
            cprint(errors, "red")
            exit(1)

        n, expected = solve(test)
        expected.sort()
        got = sorted(process_output(n, got))

        plot_n.append(n)
        plot_t.append(end - start)

        if not compare(expected, got):
            cprint("TEST FAILED", "red")
            cprint("Input:\n{}".format("\n".join([" ".join([str(el) for el in r]) for r in matrix_from_test(test)[1]])))
            cprint("Expected:", "green")
            print(expected)
            cprint("Got:", "red")
            print(got)
            exit(1)
        else:
            cprint("TEST PASSED (n = {})".format(n), "green")

    os.unlink("main")

    plot_log_n = [numpy.log(n) for n in plot_n[64:]]
    plot_log_t = [numpy.log(t) for t in plot_t[64:]]

    slope = numpy.divide(
        numpy.subtract(numpy.mean(numpy.multiply(plot_log_n, plot_log_t)), numpy.mean(plot_log_n) * numpy.mean(plot_log_t)),
        numpy.subtract(numpy.mean(numpy.square(plot_log_n)), numpy.mean(plot_log_n) ** 2),
    )

    cprint("Runtime complexity is O(N^{:.2f})".format(slope), "magenta")

    plt.xlabel("Computation time, s")
    plt.ylabel("Size of the matrix")
    plt.grid()
    plt.title("Implementation performance")
    plt.plot(plot_n, plot_t, color="r")
    plt.show()

    plt.xlabel("log(Computation time)")
    plt.ylabel("log(Size of the matrix)")
    plt.grid()
    plt.title("Implementation performance (logarithmic scale)")
    plt.plot(plot_log_n, plot_log_t, color="r", label="time (slope: {:.2f})".format(slope))
    plt.show()

test_all()
