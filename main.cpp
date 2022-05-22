#include <iostream>
#include <numeric>
#include <vector>
#include <cmath>
#include <list>


/// SOLVING CENTURY EQUATION:

static constinit double newton_eps = 1e-15;

bool is_close (const double first, const double second, const double eps = 1e-9) {
    return std::fabs(first - second) < eps;
}

double find_root_with_binary_search (
    const std::function<double(double)>& function,
    const bool increasing,
    double l,
    double r
) {
    while (r - l > newton_eps) {
        const double x = (r + l) / 2.;
        const double y = function(x);

        if (is_close(y, 0.))
            return x;

        (increasing == y > 0. ? r : l) = x;
    }

    return (r + l) / 2.;
}

double scan_for_bound (const std::function<double(double)>& function, const bool direction, const double start) {
    double current_bound = start + (direction ? 1. : -1.);

    while (function(current_bound) < 0.)
        current_bound += 2 * (current_bound - start);

    return current_bound;
}

std::vector<double> solve_century_equation (const double ro, const std::vector<std::pair<double, double>>& u2_and_d) {
    static constinit double inf = std::numeric_limits<double>::infinity();
    assert(!is_close(ro, 0.));
    const size_t n = u2_and_d.size();

    const auto function =
        [ro, &u2_and_d] (const double lambda) -> double {
            double sum = 0.;

            for (const auto& [u2, d] : u2_and_d)
                sum += u2 / (d - lambda);

            return 1. + ro * sum;
        };

    std::vector<double> roots(n);

    for (size_t i = 0; i < n; i ++) {
        double l, r;

        if (ro < 0.) {
            l = i == 0 ? scan_for_bound(function, false, u2_and_d.front().second) : u2_and_d[i - 1].second;
            r = u2_and_d[i].second;
        }

        else {
            l = u2_and_d[i].second;
            r = i < n - 1 ? u2_and_d[i + 1].second : scan_for_bound(function, true, u2_and_d.back().second);
        }

        roots[i] = find_root_with_binary_search(function, ro > 0., l, r);
    }

    return roots;
}


/// MATRICES:

class square_matrix {
public:
    explicit square_matrix (const size_t n, const double el): _n(n), _elements(n, std::vector<double>(n, el)) {}

    explicit square_matrix (const size_t n): square_matrix(n, 0.) {
        for (size_t i = 0; i < _n; i ++)
            _elements[i][i] = 1;
    }

    [[nodiscard]] const double& at (const size_t i, const size_t j) const {
        if (i > _n || j > _n)
            throw std::out_of_range("square matrix");

        return _elements[i][j];
    }

    [[nodiscard]] double& at (const size_t i, const size_t j) {
        if (i > _n || j > _n)
            throw std::out_of_range("square matrix");

        return _elements[i][j];
    }

    [[nodiscard]] size_t size () const noexcept { return _n; }

private:
    size_t _n;
    std::vector<std::vector<double>> _elements;
};

square_matrix multiply_square_matrices (const square_matrix& first, const square_matrix& second) {
    assert(first.size() == second.size());
    const size_t n = first.size();

    square_matrix res(n, 0.);

    for (size_t i = 0; i < n; i ++)
        for (size_t j = 0; j < n; j ++)
            for (size_t k = 0; k < n; k ++)
                res.at(i, j) += first.at(i, k) * second.at(k, j);

    return res;
}

class tridiagonal_matrix {
public:
    explicit tridiagonal_matrix (const std::vector<double>& elements): _n(elements.size() / 2 + 1) {
        assert(elements.size() % 2 == 1);
        const auto shift = static_cast<std::iterator_traits<std::vector<double>::const_iterator>::difference_type>(_n);
        std::copy(elements.cbegin(), std::next(elements.cbegin(), shift), std::back_inserter(_primary_diagonal));
        std::copy(std::next(elements.cbegin(), shift), elements.cend(), std::back_inserter(_secondary_diagonal));
    }

    [[nodiscard]] size_t size () const noexcept { return _n; }

    [[nodiscard]] double at (const size_t i, const size_t j) const {
        if (i >= _n || j >= _n)
            throw std::out_of_range("tridiagonal matrix: out of range");

        const size_t d = std::max(i, j) - std::min(i, j);

        switch (d) {
            case 0: return _primary_diagonal[i];
            case 1: return _secondary_diagonal[std::min(i, j)];
            default: return 0;
        }
    }

private:
    std::vector<double> _primary_diagonal;
    std::vector<double> _secondary_diagonal;
    const size_t _n;
};

template <typename matrix>
void print_matrix (const matrix& m) {
    for (size_t i = 0; i < m.size(); i ++) {
        for (size_t j = 0; j < m.size(); j ++)
            std::cout << m.at(i, j) << " ";

        std::cout << std::endl;
    }

    std::cout << "{";

    for (size_t i = 0; i < m.size(); i ++) {
        std::cout << (i != 0 ? ", " : "") << "{";

        for (size_t j = 0; j < m.size(); j ++)
            std::cout << (j != 0 ? ", " : "") << m.at(i, j);

        std::cout << "}";
    }

    std::cout << "}" << std::endl;
}


/// ORTHOGONALIZATION:

std::vector<double> scale (const std::vector<double>& vec, const double k) {
    std::vector<double> res;
    std::transform(
        vec.cbegin(),
        vec.cend(),
        std::back_inserter(res),
        [k] (const double el) -> double { return k * el; }
    );
    return res;
}

void operate_vector (
    std::vector<double>& first,
    const std::vector<double>& second,
    const std::function<double(double, double)>& f
) {
    assert(first.size() == second.size());
    const size_t n = first.size();

    for (size_t i = 0; i < n; i ++)
        first[i] = f(first[i], second[i]);
}

double dot_product (const std::vector<double>& first, const std::vector<double>& second) {
    assert(first.size() == second.size());
    const size_t n = first.size();
    double k = 0.;

    for (size_t i = 0; i < n; i ++)
        k += first[i] * second[i];

    return k;
}

double norm (const std::vector<double>& vec) {
    return std::sqrt(dot_product(vec, vec));
}

std::vector<double> project (const std::vector<double>& a, const std::vector<double>& b) {
    return scale(b, dot_product(a, b) / dot_product(b, b));
}

std::vector<std::vector<double>> orthogonalize (const std::vector<std::vector<double>>& a) {
    const size_t n = a.size();
    std::vector<std::vector<double>> b(a);

    for (size_t i = 1; i < n; i ++)
        for (size_t j = 0; j < i; j ++)
            operate_vector(b[i], project(a[i], b[j]), std::minus<>());

    for (std::vector<double>& e : b)
        e = scale(e, 1 / norm(e));

    return b;
}


/// ALGORITHM IMPL:

std::vector<std::vector<size_t>> group_d (const std::vector<double>& d) {
    std::vector<std::pair<size_t, double>> i_and_d;
    i_and_d.reserve(d.size());

    for (size_t i = 0; i < d.size(); i ++)
        i_and_d.emplace_back(i, d[i]);

    std::sort(
        i_and_d.begin(),
        i_and_d.end(),
        [] (const auto& first, const auto& second) -> bool { return first.second < second.second; }
    );

    std::vector<std::vector<size_t>> groups;

    for (const auto [i, d_val] : i_and_d) {
        auto& last_group = groups.back();

        if (!groups.empty() && is_close(d_val, d[last_group.front()]))
            last_group.push_back(i);

        else
            groups.push_back({ i });
    }

    return groups;
}

auto find_eigen_values_and_vectors (const tridiagonal_matrix& matrix) {
    const size_t n = matrix.size();

    if (n == 1)
        return std::vector<std::pair<double, std::vector<double>>>({ { matrix.at(0, 0), { 1. } } });

    /// Splitting the matrix: Q -> diag(T1, T2) + Ro * v * v^T
    const size_t m = (n - 1) / 2;
    const size_t first_size = m + 1;
    const size_t second_size = n - first_size;

    std::vector<double> first_matrix_elements;
    std::vector<double> second_matrix_elements;

    for (size_t i = 0; i < n; i ++)
        (i <= m ? first_matrix_elements : second_matrix_elements).push_back(matrix.at(i, i));

    for (size_t i = 0; i < n - 1; i ++) {
        if (i == m)
            continue;

        (i < m ? first_matrix_elements : second_matrix_elements).push_back(matrix.at(i, i + 1));
    }

    const double ro = matrix.at(m, m + 1);
    first_matrix_elements[m] -= ro;
    second_matrix_elements[0] -= ro;

    const tridiagonal_matrix first_matrix(first_matrix_elements);
    const tridiagonal_matrix second_matrix(second_matrix_elements);

    /// Build sequence d and vector u:
    const auto first_eigen_values_and_vectors = find_eigen_values_and_vectors(first_matrix);
    const auto second_eigen_values_and_vectors = find_eigen_values_and_vectors(second_matrix);

    std::vector<double> u;
    std::vector<double> d;
    u.reserve(n);
    d.reserve(n);

    for (const auto& [value, vector] : first_eigen_values_and_vectors) {
        u.push_back(vector.back());
        d.push_back(value);
    }

    for (const auto& [value, vector] : second_eigen_values_and_vectors) {
        u.push_back(vector.front());
        d.push_back(value);
    }

    /// Finding the inner matrix' eigen values and vectors:
    std::vector<double> eigen_values(n);
    eigen_values.reserve(n);
    std::vector<std::vector<double>> eigen_vectors(n, std::vector<double>(n, 0.));

    /// ro = 0 => result is trivial
    if (is_close(ro, 0.)) {
        eigen_values = d;

        for (size_t i = 0; i < n; i ++)
            eigen_vectors[i][i] = 1.;
    }

    else {
        auto groups = group_d(d);
        const auto group_d = [&d] (const auto group) -> double {
            /// the smallest d value in the group is chosen for the whole group
            return d[group.front()];
        };

        /// deflate u2 = 0:
        for (auto group_it = groups.begin(); group_it != groups.end();) {
            auto& group = *group_it;

            for (auto it = group.begin(); it != group.end();) {
                const size_t i = *it;

                if (is_close(std::pow(u[i], 2.), 0.)) {
                    eigen_values[i] = group_d(group);

                    /// ei = (0, ..., 0, 1, 0, ..., 0)^T
                    eigen_vectors[i][i] = 1.;

                    /// erase all deflated values
                    it = group.erase(it);
                }

                else
                    ++ it;
            }

            if (group.empty())
                group_it = groups.erase(group_it);

            else
                ++ group_it;
        }

        /// deflate di = dj:
        for (const auto& group : groups) {
            for (size_t t = 0; t < group.size() - 1; t ++) {
                const size_t i1 = group[t];
                const size_t i2 = group[t + 1];

                eigen_values[i1] = group_d(group);

                /// ei = (0, ..., 0, 1, 0, ..., 0, - ui1 / ui2, 0, ..., 0)^T
                eigen_vectors[i1][i1] = 1.;
                eigen_vectors[i1][i2] = - u[i1] / u[i2];
            }
        }

        /// non-deflate:
        std::vector<std::pair<double, double>> u_and_d_deflated;
        std::vector<size_t> indices; /// indices to insert new eugen values
        const size_t g = groups.size();
        u_and_d_deflated.reserve(g);
        indices.reserve(g);

        for (const auto& group : groups) {
            indices.push_back(group.back());
            u_and_d_deflated.emplace_back(0., group_d(group));

            for (const size_t i : group)
                u_and_d_deflated.back().first += std::pow(u[i], 2.);
        }

        const std::vector<double> new_eugen_values = solve_century_equation(ro, u_and_d_deflated);

        for (size_t k = 0; k < g; k ++) {
            const size_t i = indices[k];
            eigen_values[i] = new_eugen_values[k];

            for (size_t j = 0; j < n; j ++)
                eigen_vectors[i][j] = u[j] / (d[j] - new_eugen_values[k]);
        }
    }

    /// Build Qr matrix:
    square_matrix q_right(n);

    for (size_t i = 0; i < n; i ++)
        for (size_t j = 0; j < n; j ++)
            q_right.at(i, j) = eigen_vectors[j][i];

    /// Build Ql = diag(Q1, Q2) matrices:
    square_matrix q_left(n);

    for (size_t i = 0; i < first_size; i ++)
        for (size_t j = 0; j < first_size; j ++)
            q_left.at(i, j) = first_eigen_values_and_vectors[j].second[i];

    for (size_t i = 0; i < second_size; i ++)
        for (size_t j = 0; j < second_size; j ++)
            q_left.at(i + first_size, j + first_size) = second_eigen_values_and_vectors[j].second[i];

    /// building Q = Ql * Qr
    const square_matrix q = multiply_square_matrices(q_left, q_right);

    /// Building the resulting eigen vectors:
    std::vector<std::vector<double>> resulting_eigen_vectors(n, std::vector<double>(n, 0.));

    for (size_t i = 0; i < n; i ++)
        for (size_t j = 0; j < n; j ++)
            resulting_eigen_vectors[i][j] = q.at(j, i);

    resulting_eigen_vectors = orthogonalize(resulting_eigen_vectors);

    /// Result:
    std::vector<std::pair<double, std::vector<double>>> res(n);

    for (size_t i = 0; i < n; i ++)
        res[i] = std::make_pair(eigen_values[i], resulting_eigen_vectors[i]);

    return res;
}


/// DEMO:

void output (const tridiagonal_matrix& matrix, const std::vector<std::pair<double, std::vector<double>>>& res) {
    std::cout << "matrix" << std::endl;
    print_matrix(matrix);

    std::cout << "has the following eigenvalues and eigenvectors" << std::endl;
    for (const auto& [value, vector] : res)
        std::cout << value << ": (" <<
            std::accumulate(
                vector.cbegin(),
                vector.cend(),
                std::string(),
                [] (const std::string& accumulator, const double el) -> std::string {
                    return accumulator + (!accumulator.empty() ? ", " : "") + std::to_string(el);
                }
            ) << ")" << std::endl;
}

int main () {
    size_t n;
    std::cout << "Input the size of the matrix: ";
    std::cin >> n;

    if (n == 0)
        std::cout << "Size should be greater then zero!" << std::endl;

    std::vector<double> matrix_elements(2 * n - 1);
    std::cout << "Input primary diagonal: ";

    for (size_t i = 0; i < n; i ++)
        std::cin >> matrix_elements[i];

    std::cout << "Input secondary diagonal: ";

    for (size_t i = n; i < 2 * n - 1; i ++)
        std::cin >> matrix_elements[i];

    const tridiagonal_matrix matrix(matrix_elements);
    const auto res = find_eigen_values_and_vectors(matrix);
    output(matrix, res);

    return 0;
}
