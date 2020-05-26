//
// Created by nether on 22.05.2020.
//

#pragma once

#include <dlib/svm_threaded.h>

#include <iostream>
#include <vector>

#include <dlib/rand.h>

using namespace std;
using namespace dlib;

// Our data will be 2-dimensional data. So declare an appropriate type to contain these points.
const int sample_size = 8;
typedef matrix<double,sample_size,1> sample_type;
typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;
typedef polynomial_kernel<sample_type> poly_kernel;
typedef radial_basis_kernel<sample_type> rbf_kernel;

void norm(double& src, double min_, double diff)
{
    src = abs(src - min_) / diff;
}

struct stat_sample {
    stat_sample() = default;
    explicit stat_sample(const sample_type& init) : max_(init), min_(init), diff(0) {}
    void refresh(const sample_type& one) {
        for (auto i = 0; i < sample_size; ++i) {
            max_(i) = max(max_(i), one(i));
            min_(i) = min(min_(i), one(i));
        }
    }
    void normalize(sample_type& one) {
        for (auto i = 0; i < sample_size; ++i)
            diff(i) = max_(i) - min_(i);
        // x, y xoord
        norm(one(0), min_(0), diff(0));
        norm(one(1), min_(1), diff(1));
        // rooms
        norm(one(2), 0, diff(2));
        // price
        norm(one(3), min_(3), diff(3));
        // area
        norm(one(4), 0, max_(4));
        // kitchen
        norm(one(5), 0, max_(5));
        // floor
        if (round(one(6)) == 1 || round(one(6)) == round(one(7))) {
            one(6) = 1;
        } else {
            one(6) = 0;
        }
        // max_floor
        norm(one(7), 0, max_(7));
    }
    sample_type max_;
    sample_type min_;
    sample_type diff;
};

class record {
public:
    explicit record(sample_type& _sample, std::vector<sample_type>& _samples) : stat(_sample), samples(_samples) {}
    void push(sample_type& sample) {
        stat.refresh(sample);
        samples.emplace_back(sample);
    }
    void normalize() {
        noNormSamples = samples;
        for (auto& a : samples) {
            stat.normalize(a);
        }
    }
    const stat_sample& getStat(){
        return stat;
    };
    const std::vector<sample_type>& NotNormedData() const
    {
        return noNormSamples;
    }
private:
    stat_sample stat;
    std::vector<sample_type>& samples;
    std::vector<sample_type> noNormSamples;
};


void parseString(std::string& inp_str, sample_type& inp)
{
    try {
        size_t i = 0;
        while (inp_str.find(';') != inp_str.npos) {
            inp(i) = inp_str.find(';') != 0 ?
                            atof(inp_str.substr(0, inp_str.find(';')).c_str()) : 0;
            inp_str = inp_str.substr(inp_str.find(';') + 1, inp_str.size() - inp_str.find(';')).c_str();
            ++i;
        }
        inp(i) = atof(inp_str.c_str());
    } catch (std::exception& ex) {
        std::cout << "error in parsing input string" << std::endl;
        std::cout << ex.what() << std::endl;
        throw;
    }
}