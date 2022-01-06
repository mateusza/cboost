//
// Copyright 2021-2022 Mateusz Adamowski <mateusz at adamowski dot pl>
//
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <string>

#ifndef CBOOST_HPP_
#define CBOOST_HPP_

/* simple re-implementation of some python builtin methods */

namespace py {

namespace methods {

template <typename T> std::string join(const std::string g, const T elts);
std::vector<std::string> split(std::string st, const std::string sp);
template <typename T> void extend(std::vector<T>&, const std::vector<T>);

}  // namespace methods


namespace classes {

template <typename T> struct range;

}  // namespace classes

}  // namespace py

/* simple re-implementation of some python builtin functions */
template <typename T>
bool all(const T elts) {
    for (auto e : elts)
        if (!e)
            return false;
    return true;
}

template <typename T>
bool any(const T elts) {
    for (auto e : elts)
        if (e)
            return true;
    return false;
}

template <typename T>
T sum(const std::vector<T> elts) {
    T s{0};
    for (auto e : elts)
        s += e;
    return s;
}

template <typename T>
std::string str(const T n) {
    return std::to_string(n);
}

template <>
std::string str(const std::string s) {
    return s;
}

template <>
std::string str(const char *s) {
    return std::string{s};
}

template <>
std::string str(const char s) {
    return std::string{s};
}

template <>
std::string str(const bool b) {
    return b?"True":"False";
}

template<typename VT>
std::string str(std::vector<VT> vec) {
    return "["+py::methods::join(", ", vec)+"]";
}

template<typename T>
void print(const T v) {
    std::cout << str(v) << std::endl;
}

template<typename T>
int64_t len(const T c) {
    throw std::runtime_error("len() not supported with this type");
}

template<>
int64_t len(const std::string s) {
    return s.length();
}

template<typename VT>
int64_t len(const std::vector<VT> v) {
    return v.size();
}

template<typename T>
int64_t len(const py::classes::range<T> r) {
    return r.length();
}

template<typename T>
std::string operator * (const T& n, const std::string s) {
    std::stringstream r;
    for (auto i=0; i < n; ++i)
        r << s;
    return r.str();
}

template <typename T>
std::string operator * (const std::string s, const T& n) {
    return n * s;
}

template <typename T>
std::vector<T> operator + (
        const std::vector<T> &lhs,
        const std::vector<T> &rhs) {
    std::vector<T> r{};
    r.reserve(lhs.size() + rhs.size());
    py::methods::extend(r, lhs);
    py::methods::extend(r, rhs);
    return r;
}

template <typename T>
T abs(const T n) {
    return (n < 0) ? -n : n;
}

template<typename T, typename VT>
std::vector<VT> operator * (const T& n, const std::vector<VT> v) {
    std::vector<VT> r{};
    if (v.size() == 1) {
        r.reserve(n);
        std::fill_n(r.begin(), n, v[0]);
        return r;
    }
    r.reserve(v.size() * n);
    for (auto i=0; i < n; ++i)
        py::methods::extend(r, v);
    return r;
}

template<typename T, typename VT>
std::vector<VT> operator * (const std::vector<VT> v, const T& n) {
    return n * v;
}

template<typename T>
py::classes::range<T> range(T a) {
    return py::classes::range<T>{a};
}

template<typename T, typename T2>
py::classes::range<T> range(T a, T2 b) {
    return {a, static_cast<T>(b)};
}

template<typename T, typename T2, typename T3>
py::classes::range<T> range(T a, T2 b, T3 c) {
    return {a, static_cast<T>(b), static_cast<T>(c)};
}

/* simple re-implementation of some python builtin types' methods */
namespace py {

namespace operators {

    template <typename T>
    std::vector<T> starred(const py::classes::range<T>& ra) {
        std::vector<T> r{};
        r.reserve(ra.length());
        for (auto i : ra){
            r.push_back(i);
        }
        return r;
    }

    template <typename T>
    std::vector<T> starred(const std::vector<T>& sv) {
        std::vector<T> r{};
        r.reserve(sv.size());
        std::copy(sv.begin(), sv.end(), std::back_inserter(r));
        return r;
    }

    std::vector<std::string> starred(const std::string& s) {
        std::vector<std::string> r{};
        r.reserve(s.size());
        for (auto c : s){
            r.push_back(std::string{c});
        }
        return r;
    }

}  // namespace operators

namespace methods {

template<typename T>
std::string join(const std::string g, const T elts) {
    std::stringstream r; bool c{0};
    for (auto e : elts) {
        if (c)
            r << g;
        r << str(e);
        c = 1;
    }
    return r.str();
}

std::vector<std::string> split(std::string st, const std::string sp) {
    std::vector<std::string> r{};
    int end;
    while (true) {
        end = st.find(sp);
        if (end == std::string::npos) {
            r.push_back(st);
            break;
        }
        r.push_back(st.substr(0, end));
        st = st.substr(end + sp.size());
    }
    return r;
}

template <typename T>
void append(std::vector<T>& l, const T e) { // NOLINT
    l.push_back(e);
}

template <typename T>
void extend(std::vector<T>& l, const std::vector<T> src) { // NOLINT
    std::copy(src.begin(), src.end(), std::back_inserter(l));
}

}  // namespace methods

namespace classes {
template <typename T>
struct range {
    struct iterator {
        iterator(const T val, const T step, const T stop):
            current{val}, step{step}, stop{stop} {}
        T& operator*() { return current; }
        iterator& operator++() {
            current += step;
            if ((step > 0 && current > stop) || (step < 0 && current < stop))
                current = stop;
            return *this;
        }
        friend bool operator== (const iterator& a, const iterator& b) {
            return a.current == b.current;
        }
        friend bool operator!= (const iterator& a, const iterator& b) {
            return a.current != b.current;
        }

     private:
        T current, step, stop;
    };
    iterator begin() const {
        auto pos = ((step < 0 && start < stop) || (step > 0 && stop < start))
            ? this->stop : this->start;
        return {pos, this->step, this->stop};
    }
    iterator end() const {
        return {this->stop, this->step, this->stop};
    }
    explicit range(const T stop):
        range{0, stop} {};
    range(const T start, T stop):
        range{start, stop, 1} {};
    range(const T start, T stop, T step):
        start{start},
        stop{stop},
        step{step}
    {
        if (step == 0)
            throw std::runtime_error("range() step cannot be zero");
    }
    int64_t length() const {
        return (stop - start) / step;
    }

 private:
    T start, stop, step;
};
}  // namespace classes

}  // namespace py

#endif  // CBOOST_HPP_

