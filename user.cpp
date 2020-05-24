//
// Created by nether on 22.05.2020.
//

#include "classi.h"

double cheb_distance(const sample_type& a, const sample_type& b) {
    double res = 0;
    for (auto i = 0; i < sample_size; ++i) {
        double diff = abs(a(i) - b(i));
        res = max(res, diff);
    }
    return res;
}

struct OutPoints{
    explicit OutPoints(const sample_type& init, const sample_type& initNN, const sample_type& orient) : smp(init), smpNN(initNN) {
        dst = cheb_distance(smp, orient);
    }
    sample_type smp;
    sample_type smpNN;
    double dst;
};

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cout << "Usage: rclss model_file_name" << std::endl;
        return 1;
    }
    std::vector<sample_type> samples;
    std::vector<sample_type> notNormSamples;
    std::vector<double> labels;
    // If you want to save a one_vs_one_decision_function to disk, you can do
    // so.  However, you must declare what kind of decision functions it contains.
    one_vs_one_decision_function<ovo_trainer,
            decision_function<poly_kernel>,  // This is the output of the poly_trainer
            decision_function<rbf_kernel>    // This is the output of the rbf_trainer
    > df3;
    // load the function back in from disk and store it in df3.
    stat_sample stats;
    deserialize(argv[1]) >> df3 >> notNormSamples >> samples >> labels >> stats.min_ >> stats.max_;

    if (labels.size() != samples.size()) {
        std::cout << "error in read" << std::endl;
        return 1;
    }

    sample_type inp;
    std::string inp_str;
    std::cin >> inp_str;
    try {
        size_t i = 0;
        while (inp_str.find(';') != inp_str.npos) {
            inp(i) = atof(inp_str.substr(0, inp_str.find(';')).c_str());
            inp_str = inp_str.substr(inp_str.find(';') + 1, inp_str.size() - inp_str.find(';')).c_str();
            ++i;
        }
        inp(i) = atof(inp_str.c_str());
    } catch (std::exception ex) {
        std::cout << "error in parsing input string" << std::endl;
        std::cout << ex.what() << std::endl;
        return 1;
    }
   // sample_type inpNoNorm = inp;
    stats.normalize(inp);
    std::vector<OutPoints> out;
    int res = df3(inp);
    for (auto i = 0; i < labels.size(); ++i) {
        if (round(labels[i]) == res) {
            out.emplace_back(samples[i], notNormSamples[i], inp);
        }
    }
    for(const auto& a : out)
    std::sort(out.begin(), out.end(), [](const OutPoints& a, const OutPoints& b) {
        return a.dst < b.dst;
    });

    for(const auto& a : out) {
        for (auto i = 0; i < sample_size; ++i)
            std::cout << a.smpNN(i) << " ";
        std::cout << std::endl;
    }
}