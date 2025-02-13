#pragma once

#include <array>

#define ExtensionTest_VEC_TEMPLATE_CONTENT(T, N) \
  public: \
    using Base = std::array<T, N>; \
 \
    Vec() : Base() { \
        std::fill(this->begin(), this->end(), T()); \
    } \
 \
    explicit(false) Vec(Base array) : Base(std::move(array)) {} \
 \
    static Vec Full(const T &value) { \
        Vec ret {}; \
        std::fill(ret.begin(), ret.end(), value); \
        return ret; \
    }

namespace ExtensionTest {
    template<typename T, std::size_t N>
    class Vec : public std::array<T, N> {
        ExtensionTest_VEC_TEMPLATE_CONTENT(T, N)
    };

    template<>
    class Vec<float, 2> : public std::array<float, 2> {
        ExtensionTest_VEC_TEMPLATE_CONTENT(float, 2)

    public:
        Vec(float x, float y) : Base({x, y}) {
        }

        [[nodiscard]] const float &X() const { return at(0); }
        [[nodiscard]] const float &Y() const { return at(1); }

        [[nodiscard]] std::string Repr() const {
            return "<Vec2f: (" + std::to_string(X()) + ", " + std::to_string(Y()) + ")>";
        }
    };

    template<>
    class Vec<int, 2> : public std::array<int, 2> {
        ExtensionTest_VEC_TEMPLATE_CONTENT(int, 2)

    public:
        Vec(int x, int y) : Base({x, y}) {
        }

        [[nodiscard]] const int &X() const { return at(0); }
        [[nodiscard]] const int &Y() const { return at(1); }

        [[nodiscard]] std::string Repr() const {
            return "<Vec2f: (" + std::to_string(X()) + ", " + std::to_string(Y()) + ")>";
        }
    };

    template<>
    class Vec<float, 3> : public std::array<float, 3> {
        ExtensionTest_VEC_TEMPLATE_CONTENT(float, 3)

    public:
        Vec(float x, float y, float z) : Base({x, y, z}) {
        }

        [[nodiscard]] const float &X() const { return at(0); }
        [[nodiscard]] const float &Y() const { return at(1); }
        [[nodiscard]] const float &Z() const { return at(2); }

        [[nodiscard]] std::string Repr() const {
            return "<Vec2f: (" + std::to_string(X()) + ", " + std::to_string(Y()) + ", " + std::to_string(Z()) + ")>";
        }
    };

    template<>
    class Vec<int, 3> : public std::array<int, 3> {
        ExtensionTest_VEC_TEMPLATE_CONTENT(int, 3)

    public:
        Vec(int x, int y, int z) : Base({x, y, z}) {
        }

        [[nodiscard]] const int &X() const { return at(0); }
        [[nodiscard]] const int &Y() const { return at(1); }
        [[nodiscard]] const int &Z() const { return at(2); }

        [[nodiscard]] std::string Repr() const {
            return "<Vec2f: (" + std::to_string(X()) + ", " + std::to_string(Y()) + ", " + std::to_string(Z()) + ")>";
        }
    };

    inline void VecPythonBindings(const pybind11::module_ &m) {
        pybind11::class_<Vec<float, 2> >(m, "Vec2f") //
                .def(pybind11::init<float, float>()) //
                .def("x", &Vec<float, 2>::X) //
                .def("y", &Vec<float, 2>::Y) //
                .def("__repr__", &Vec<float, 2>::Repr);
        pybind11::class_<Vec<int, 2> >(m, "Vec2i") //
                .def(pybind11::init<int, int>()) //
                .def("x", &Vec<int, 2>::X) //
                .def("y", &Vec<int, 2>::Y) //
                .def("__repr__", &Vec<int, 2>::Repr);
        pybind11::class_<Vec<float, 3> >(m, "Vec3f") //
                .def(pybind11::init<float, float, float>()) //
                .def("x", &Vec<float, 3>::X) //
                .def("y", &Vec<float, 3>::Y) //
                .def("z", &Vec<float, 3>::Z) //
                .def("__repr__", &Vec<float, 3>::Repr);
        pybind11::class_<Vec<int, 3> >(m, "Vec3i") //
                .def(pybind11::init<int, int, int>()) //
                .def("x", &Vec<int, 3>::X) //
                .def("y", &Vec<int, 3>::Y) //
                .def("z", &Vec<int, 3>::Z) //
                .def("__repr__", &Vec<int, 3>::Repr);
    }

    // Element-wise function:

    template<typename newT, typename oldT, std::size_t N>
    Vec<newT, N> VecApply(const std::function<newT(oldT)> &f, const Vec<oldT, N> &v) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<newT, N> {
            return {{f(v[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename newT, typename oldT, std::size_t N>
    Vec<newT, N> VecApply(newT (*f)(oldT), const Vec<oldT, N> &v) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<newT, N> {
            return {{f(v[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise addition: +

    template<typename T, std::size_t N>
    Vec<T, N> operator+(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] + rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<T, N> operator+(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] + rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<T, N> operator+(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs + rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise subtraction: -

    template<typename T, std::size_t N>
    Vec<T, N> operator-(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] - rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<T, N> operator-(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] - rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<T, N> operator-(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs - rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise multiplication: *

    template<typename T, std::size_t N>
    Vec<T, N> operator*(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] * rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<T, N> operator*(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] * rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<T, N> operator*(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs * rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise division: /

    template<typename T, std::size_t N>
    Vec<T, N> operator/(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] / rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<T, N> operator/(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] / rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<T, N> operator/(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs / rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Dot product

    template<typename T, std::size_t N>
    T VecDot(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> T {
            return ((lhs[indices] * rhs[indices]) + ...);
        }(std::make_index_sequence<N>{});
    }

    // Static cast

    template<typename newT, typename oldT, std::size_t N>
    Vec<newT, N> VecCast(const Vec<oldT, N> &vec) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<newT, N> {
            return {{static_cast<newT>(vec[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Boolean all

    template<std::size_t N>
    bool VecAll(const Vec<bool, N> &v) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> bool {
            return (v[indices] && ...);
        }(std::make_index_sequence<N>{});
    }

    // Boolean none

    template<std::size_t N>
    bool VecNone(const Vec<bool, N> &v) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> bool {
            return (!v[indices] && ...);
        }(std::make_index_sequence<N>{});
    }

    // Element-wise greater-than: >

    template<typename T, std::size_t N>
    Vec<bool, N> operator>(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] > rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator>(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] > rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator>(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs > rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise greater-or-equal-than: >=

    template<typename T, std::size_t N>
    Vec<bool, N> operator>=(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] >= rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator>=(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] >= rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator>=(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs >= rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise less-than: <

    template<typename T, std::size_t N>
    Vec<bool, N> operator<(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] < rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator<(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] < rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator<(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs < rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise less--or-equal-than: <=

    template<typename T, std::size_t N>
    Vec<bool, N> operator<=(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] <= rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator<=(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs[indices] <= rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator<=(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{(lhs <= rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }
} // namespace ExtensionTest
