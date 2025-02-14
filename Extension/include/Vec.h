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
    Vec(Base array) : Base(std::move(array)) {} \
 \
    static Vec Full(const T &value) { \
        Vec ret {}; \
        std::fill(ret.begin(), ret.end(), value); \
        return ret; \
    }

namespace ExtensionTest {
    template<typename T, std::size_t N>
    class Vec : public std::array<T, N>, public torch::CustomClassHolder {
        ExtensionTest_VEC_TEMPLATE_CONTENT(T, N)
    };

    template<>
    class Vec<double, 2> : public std::array<double, 2>, public torch::CustomClassHolder {
        ExtensionTest_VEC_TEMPLATE_CONTENT(double, 2)

    public:
        Vec(double x, double y) : Base({x, y}) {
        }

        [[nodiscard]] const double &X() const { return at(0); }
        [[nodiscard]] double &X() { return at(0); }
        [[nodiscard]] const double &Y() const { return at(1); }
        [[nodiscard]] double &Y() { return at(1); }

        [[nodiscard]] std::string Repr() const {
            return "<Vec2f: (" + std::to_string(X()) + ", " + std::to_string(Y()) + ")>";
        }
    };

    template<>
    class Vec<int64_t, 2> : public std::array<int64_t, 2>, public torch::CustomClassHolder {
        ExtensionTest_VEC_TEMPLATE_CONTENT(int64_t, 2)

    public:
        Vec(int64_t x, int64_t y) : Base({x, y}) {
        }

        [[nodiscard]] const int64_t &X() const { return at(0); }
        [[nodiscard]] int64_t &X() { return at(0); }
        [[nodiscard]] const int64_t &Y() const { return at(1); }
        [[nodiscard]] int64_t &Y() { return at(1); }

        [[nodiscard]] std::string Repr() const {
            return "<Vec2f: (" + std::to_string(X()) + ", " + std::to_string(Y()) + ")>";
        }
    };

    template<>
    class Vec<double, 3> : public std::array<double, 3>, public torch::CustomClassHolder {
        ExtensionTest_VEC_TEMPLATE_CONTENT(double, 3)

    public:
        Vec(double x, double y, double z) : Base({x, y, z}) {
        }

        [[nodiscard]] const double &X() const { return at(0); }
        [[nodiscard]] double &X() { return at(0); }
        [[nodiscard]] const double &Y() const { return at(1); }
        [[nodiscard]] double &Y() { return at(1); }
        [[nodiscard]] const double &Z() const { return at(2); }
        [[nodiscard]] double &Z() { return at(2); }

        [[nodiscard]] std::string Repr() const {
            return "<Vec2f: (" + std::to_string(X()) + ", " + std::to_string(Y()) + ", " + std::to_string(Z()) + ")>";
        }
    };

    template<>
    class Vec<int64_t, 3> : public std::array<int64_t, 3>, public torch::CustomClassHolder {
        ExtensionTest_VEC_TEMPLATE_CONTENT(int64_t, 3)

    public:
        Vec(int64_t x, int64_t y, int64_t z) : Base({x, y, z}) {
        }

        [[nodiscard]] const int64_t &X() const { return at(0); }
        [[nodiscard]] int64_t &X() { return at(0); }
        [[nodiscard]] const int64_t &Y() const { return at(1); }
        [[nodiscard]] int64_t &Y() { return at(1); }
        [[nodiscard]] const int64_t &Z() const { return at(2); }
        [[nodiscard]] int64_t &Z() { return at(2); }

        [[nodiscard]] std::string Repr() const {
            return "<Vec2f: (" + std::to_string(X()) + ", " + std::to_string(Y()) + ", " + std::to_string(Z()) + ")>";
        }
    };

    inline void VecPythonBindings(torch::Library &m) {
        m.class_<Vec<double, 2> >("Vec2f") //
                .def(torch::init<double, double>()) //
                .def("x", static_cast<const double &(Vec<double, 2>::*)() const>(&Vec<double, 2>::X)) //
                .def("y", static_cast<const double &(Vec<double, 2>::*)() const>(&Vec<double, 2>::Y)) //
                .def("__repr__", &Vec<double, 2>::Repr);
        m.class_<Vec<int64_t, 2> >("Vec2i") //
                .def(torch::init<int64_t, int64_t>()) //
                .def("x", static_cast<const int64_t &(Vec<int64_t, 2>::*)() const>(&Vec<int64_t, 2>::X)) //
                .def("y", static_cast<const int64_t &(Vec<int64_t, 2>::*)() const>(&Vec<int64_t, 2>::Y)) //
                .def("__repr__", &Vec<int64_t, 2>::Repr);
        m.class_<Vec<double, 3> >("Vec3f") //
                .def(torch::init<double, double, double>()) //
                .def("x", static_cast<const double &(Vec<double, 3>::*)() const>(&Vec<double, 3>::X)) //
                .def("y", static_cast<const double &(Vec<double, 3>::*)() const>(&Vec<double, 3>::Y)) //
                .def("z", static_cast<const double &(Vec<double, 3>::*)() const>(&Vec<double, 3>::Z)) //
                .def("__repr__", &Vec<double, 3>::Repr);
        m.class_<Vec<int64_t, 3> >("Vec3i") //
                .def(torch::init<int64_t, int64_t, int64_t>()) //
                .def("x", static_cast<const int64_t &(Vec<int64_t, 3>::*)() const>(&Vec<int64_t, 3>::X)) //
                .def("y", static_cast<const int64_t &(Vec<int64_t, 3>::*)() const>(&Vec<int64_t, 3>::Y)) //
                .def("z", static_cast<const int64_t &(Vec<int64_t, 3>::*)() const>(&Vec<int64_t, 3>::Z)) //
                .def("__repr__", &Vec<int64_t, 3>::Repr);
    }

    // Convert between PyTorch int array:

    template<typename intT, std::size_t N>
    at::IntArrayRef VecToIntArrayRef(const Vec<intT, N> &v) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> at::IntArrayRef {
            return at::IntArrayRef({v[indices]...});
        }(std::make_index_sequence<N>{});
    }

    template<typename intT, std::size_t N>
    Vec<intT, N> VecFromIntArrayRef(const at::IntArrayRef &v) {
        assert(v.size() == N);
        Vec<intT, N> ret{};
        int index = 0;
        for (int64_t e: v) {
            ret[index++] = static_cast<intT>(e);
        }
        return std::move(ret);
    }

    // Flip element order:

    template<typename T, std::size_t N>
    Vec<T, N> VecFlip(const Vec<T, N> &v) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<T, N> {
            return {{v[N - 1 - indices]...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise function:

    template<typename newT, typename oldT, std::size_t N>
    Vec<newT, N> VecApply(const std::function<newT(oldT)> &f, const Vec<oldT, N> &v) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<newT, N> {
            return {{f(v[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename newT, typename oldT, std::size_t N>
    Vec<newT, N> VecApply(const std::function<newT(const oldT &)> &f, const Vec<oldT, N> &v) {
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

    template<typename newT, typename oldT, std::size_t N>
    Vec<newT, N> VecApply(newT (*f)(const oldT &), const Vec<oldT, N> &v) {
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

    // Sum

    template<typename T, std::size_t N>
    T VecSum(const Vec<T, N> &v) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> T {
            return (v[indices] + ...);
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
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs[indices] > rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator>(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs[indices] > rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator>(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs > rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise greater-or-equal-than: >=

    template<typename T, std::size_t N>
    Vec<bool, N> operator>=(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs[indices] >= rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator>=(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs[indices] >= rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator>=(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs >= rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise less-than: <

    template<typename T, std::size_t N>
    Vec<bool, N> operator<(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs[indices] < rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator<(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs[indices] < rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator<(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs < rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    // Element-wise less--or-equal-than: <=

    template<typename T, std::size_t N>
    Vec<bool, N> operator<=(const Vec<T, N> &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs[indices] <= rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator<=(const Vec<T, N> &lhs, const T &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs[indices] <= rhs)...}};
        }(std::make_index_sequence<N>{});
    }

    template<typename T, std::size_t N>
    Vec<bool, N> operator<=(const T &lhs, const Vec<T, N> &rhs) {
        return [&]<std::size_t... indices>(std::index_sequence<indices...>) -> std::array<bool, N> {
            return {{(lhs <= rhs[indices])...}};
        }(std::make_index_sequence<N>{});
    }
} // namespace ExtensionTest
