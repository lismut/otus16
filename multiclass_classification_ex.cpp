// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the multiclass classification tools  
    from the dlib C++ Library.  Specifically, this example will make points from 
    three classes and show you how to train a multiclass classifier to recognize 
    these three classes.

    The classes are as follows:
        - class 1: points very close to the origin
        - class 2: points on the circle of radius 10 around the origin
        - class 3: points that are on a circle of radius 4 but not around the origin at all
*/
#include <string_view>
#include "classi.h"
#include "csv/csv.h"

// ----------------------------------------------------------------------------------------
void read_data (
        std::vector<sample_type>& samples,
        std::vector<double>& labels,
        const std::string& filename
)
{

}

/*!
    ensures
        - make some 3 class data as described above.  
        - Create 60 points from class 1
        - Create 70 points from class 2
        - Create 80 points from class 3
!*/
void clusterize(std::vector<sample_type>& samples,
                std::vector<double>& labels,
                int number_of_clusters)
{
    // Here we declare that our samples will be 2 dimensional column vectors.
    // (Note that if you don't know the dimensionality of your vectors at compile time
    // you can change the 2 to a 0 and then set the size at runtime)

    // Now we are making a typedef for the kind of kernel we want to use.  I picked the
    // radial basis kernel because it only has one parameter and generally gives good
    // results without much fiddling.
    typedef radial_basis_kernel<sample_type> kernel_type;


    // Here we declare an instance of the kcentroid object.  It is the object used to
    // represent each of the centers used for clustering.  The kcentroid has 3 parameters
    // you need to set.  The first argument to the constructor is the kernel we wish to
    // use.  The second is a parameter that determines the numerical accuracy with which
    // the object will perform part of the learning algorithm.  Generally, smaller values
    // give better results but cause the algorithm to attempt to use more dictionary vectors
    // (and thus run slower and use more memory).  The third argument, however, is the
    // maximum number of dictionary vectors a kcentroid is allowed to use.  So you can use
    // it to control the runtime complexity.
    kcentroid<kernel_type> kc(kernel_type(0.001),0.006, 200/(number_of_clusters*8));

    // Now we make an instance of the kkmeans object and tell it to use kcentroid objects
    // that are configured with the parameters from the kc object we defined above.
    kkmeans<kernel_type> test(kc);

    std::vector<sample_type> initial_centers;

    // tell the kkmeans object we made that we want to run k-means with k set to 3.
    // (i.e. we want 3 clusters)
    test.set_number_of_centers(number_of_clusters);

    // You need to pick some initial centers for the k-means algorithm.  So here
    // we will use the dlib::pick_initial_centers() function which tries to find
    // n points that are far apart (basically).
    pick_initial_centers(number_of_clusters, initial_centers, samples, test.get_kernel());

    // now run the k-means algorithm on our set of samples.
    test.train(samples,initial_centers);
    for (auto& a : samples) {
        labels.push_back(test(a));
    }

}
// ----------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    try
    {

        if (argc != 3) {
            std::cout << "Usage: rclst clasters_count model_file_name" << std::endl;
            return 1;
        }
        int cluster_count = 0;
        cluster_count = atoi(argv[1]);
        std::vector<sample_type> samples;
        std::vector<double> labels;

        // First, get our labeled set of training data
        std::string filename = "dataset.csv";

        io::CSVReader<8, io::trim_chars<>, io::no_quote_escape<';'>> in(filename);
        //in.read_header(io::ignore_missing_column, "x", "y", "rooms", "price", "area", "kitchen", "floor", "max_floor");
        double x(0), y(0), rooms(0), price(0), area(0), kitchen(0), floor(0), max_floor(0);
        in.read_row(x, y, rooms, price, area, kitchen, floor, max_floor);
        record rec({x, y, rooms, price, area, kitchen, floor, max_floor},
                   samples);
        while (in.read_row(x, y, rooms, price, area, kitchen, floor, max_floor)) {
            rec.push({x, y, rooms, price, area, kitchen, floor, max_floor});
        }
        rec.normalize();

        clusterize(samples, labels, cluster_count);
        cout << "samples.size(): "<< samples.size() << endl;
        cout << "labels.size(): "<< labels.size() << endl;
        if (samples.empty() || labels.empty()) {
            std::cout << "Zero size, exit" << std::endl;
            return 1;
        }
        // The main object in this example program is the one_vs_one_trainer.  It is essentially 
        // a container class for regular binary classifier trainer objects.  In particular, it 
        // uses the any_trainer object to store any kind of trainer object that implements a 
        // .train(samples,labels) function which returns some kind of learned decision function.  
        // It uses these binary classifiers to construct a voting multiclass classifier.  If 
        // there are N classes then it trains N*(N-1)/2 binary classifiers, one for each pair of 
        // labels, which then vote on the label of a sample.
        //
        // In this example program we will work with a one_vs_one_trainer object which stores any 
        // kind of trainer that uses our sample_type samples.



        // Finally, make the one_vs_one_trainer.
        ovo_trainer trainer;


        // Next, we will make two different binary classification trainer objects.  One
        // which uses kernel ridge regression and RBF kernels and another which uses a
        // support vector machine and polynomial kernels.  The particular details don't matter.
        // The point of this part of the example is that you can use any kind of trainer object
        // with the one_vs_one_trainer.


        // make the binary trainers and set some parameters
        krr_trainer<rbf_kernel> rbf_trainer;
        svm_nu_trainer<poly_kernel> poly_trainer;
        poly_trainer.set_kernel(poly_kernel(0.1, 1, 2));
        rbf_trainer.set_kernel(rbf_kernel(0.1));


        // Now tell the one_vs_one_trainer that, by default, it should use the rbf_trainer
        // to solve the individual binary classification subproblems.
        trainer.set_trainer(rbf_trainer);
        // We can also get more specific.  Here we tell the one_vs_one_trainer to use the
        // poly_trainer to solve the class 1 vs class 2 subproblem.  All the others will
        // still be solved with the rbf_trainer.
        trainer.set_trainer(poly_trainer, 1, 2);

        // Now let's do 5-fold cross-validation using the one_vs_one_trainer we just setup.
        // As an aside, always shuffle the order of the samples before doing cross validation.  
        // For a discussion of why this is a good idea see the svm_ex.cpp example.
        randomize_samples(samples, labels);
        cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
        // The output is shown below.  It is the confusion matrix which describes the results.  Each row 
        // corresponds to a class of data and each column to a prediction.  Reading from top to bottom, 
        // the rows correspond to the class labels if the labels have been listed in sorted order.  So the
        // top row corresponds to class 1, the middle row to class 2, and the bottom row to class 3.  The
        // columns are organized similarly, with the left most column showing how many samples were predicted
        // as members of class 1.
        // 
        // So in the results below we can see that, for the class 1 samples, 60 of them were correctly predicted
        // to be members of class 1 and 0 were incorrectly classified.  Similarly, the other two classes of data
        // are perfectly classified.
        /*
            cross validation: 
            60  0  0 
            0 70  0 
            0  0 80 
        */

        // Next, if you wanted to obtain the decision rule learned by a one_vs_one_trainer you 
        // would store it into a one_vs_one_decision_function.
        one_vs_one_decision_function<ovo_trainer> df = trainer.train(samples, labels);

        cout << "predicted label: "<< df(samples[0])  << ", true label: "<< labels[0] << endl;
        cout << "predicted label: "<< df(samples[90]) << ", true label: "<< labels[90] << endl;
        // The output is:
        /*
            predicted label: 2, true label: 2
            predicted label: 1, true label: 1
        */


        // If you want to save a one_vs_one_decision_function to disk, you can do
        // so.  However, you must declare what kind of decision functions it contains.
        one_vs_one_decision_function<ovo_trainer,
                                    decision_function<poly_kernel>,  // This is the output of the poly_trainer
                                    decision_function<rbf_kernel>    // This is the output of the rbf_trainer
                                    > df2;
        // Put df into df2 and then save df2 to disk.  Note that we could have also said
        // df2 = trainer.train(samples, labels);  But doing it this way avoids retraining.
        df2 = df;
        serialize(argv[2]) << df2 << rec.NotNormedData() << samples << labels << rec.getStat().min_ << rec.getStat().max_;
        // Finally, if you want to get the binary classifiers from inside a multiclass decision
        // function you can do it by calling get_binary_decision_functions() like so:
        one_vs_one_decision_function<ovo_trainer>::binary_function_table functs;
        functs = df.get_binary_decision_functions();
        cout << "number of binary decision functions in df: " << functs.size() << endl;

        // The functs object is a std::map which maps pairs of labels to binary decision
        // functions.  So we can access the individual decision functions like so:
        decision_function<poly_kernel> df_1_2 = any_cast<decision_function<poly_kernel> >(functs[make_unordered_pair(1,2)]);
        decision_function<rbf_kernel>  df_1_3 = any_cast<decision_function<rbf_kernel>  >(functs[make_unordered_pair(1,3)]);
        // df_1_2 contains the binary decision function that votes for class 1 vs. 2.
        // Similarly, df_1_3 contains the classifier that votes for 1 vs. 3.

        // Note that the multiclass decision function doesn't know what kind of binary
        // decision functions it contains.  So we have to use any_cast to explicitly cast
        // them back into the concrete type.  If you make a mistake and try to any_cast a
        // binary decision function into the wrong type of function any_cast will throw a
        // bad_any_cast exception.

    }
    catch (std::exception& e)
    {
        cout << "exception thrown!" << endl;
        cout << e.what() << endl;
    }
}


