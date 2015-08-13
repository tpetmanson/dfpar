# Extremly simple and minimalistic C++ decision forests library

Welcome to the page of DFPAR library, extremly simple and minimalistic
C++ decision forests library. The library implements decision forest building,
classification and serialization. The building and classification can use
number of cores of a single machine, thus reducing the time to build larger
models on machines with multiple cores. There are numerous decision forest
implementation available on the web, but none of them seem to fulfill all
the following requirements (as of September 2012):

* Multithreading support.
* Available as a simple header-only library.
* Depends only on C++11 standard library.
* Easy serialization/deserialization.
* DFPAR is written using C++11 threading and computing facilities and has
* no dependencies other than the standard library.

However, as most current C++ compilers do not yet fully support C++11
standard, DFPAR can be enabled to use Boost threading and random number
libraries instead of standard ones.

## License

The MIT License (MIT)

Copyright (c) 2015 Timo Petmanson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


# Example usage

```
#include "dfpar.hpp"
using namespace dfpar;

// define the number of cores we want to use for training and classification.
const size_t NUM_CORES = 4;

void demo() {
    // Declare a dataset of float type features and string type labels.
    // You can use most numeric (including discrete like integers) for
    // features and even more data types for labels, as long as they
    // can be reasonably compared. For custom datatypes, you need to
    // to hack the library.
    DataSet<double, std::string> data;

    // load the dataset from file
    {
        std::ifstream fin("data/iris.data");
        fin >> data;
        fin.close();
    }

    {
        // Initiate the decision forest builder with 500 trees
        // and two random features to try at each split.
        DForestBuilder<double, std::string> builder(data);
        builder.setNumTrees(500);
        // I would suggest number of random features from 2 up to sqrt(num_features),
        // test and see which one works best on your data.
        builder.setNumRandomFeatures(2);

        // build the forest
        DForest<double, std::string> dforest = builder.build(NUM_CORES);

        // report the out-of-bag (OOB) error
        std::cout << "Oob error: " << dforest.getOobError() << std::endl;

        // store the tree on the disk
        std::ofstream fout("model");
        fout << dforest;
        fout.close();
    }

    // load the model back from the disk
    DForest<double, std::string> dforest;
    std::ifstream fin("model");
    fin >> dforest;
    fin.close();

    // classify the same dataset with the built model
    // (this is something you do not do, when doing real evaluation)
    std::vector<ResultRow<std::string>> results = dforest.classifyAll(data, NUM_CORES);
    size_t numCorrect = 0;
    for (auto iter=results.begin() ; iter != results.end() ; ++iter) {
        numCorrect += (*iter).getResponse() == data.getY(iter-results.begin());
    }
    // print the number of correctly classified instances
    std::cerr << "Classified " << numCorrect << " correct out from ";
    std::cerr << data.numRows() << std::endl;
}

int main(int argc, char **argv) {
    try {   
        demo();
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unkown exception!" << std::endl;
    }
    return EXIT_SUCCESS;
}
```