/*
 *    _____            _     _                  __                    _   
 *   |  __ \          (_)   (_)                / _|                  | |  
 *   | |  | | ___  ___ _ ___ _  ___  _ __     | |_ ___  _ __ ___  ___| |_ 
 *   | |  | |/ _ \/ __| / __| |/ _ \| '_ \    |  _/ _ \| '__/ _ \/ __| __|
 *   | |__| |  __/ (__| \__ \ | (_) | | | |   | || (_) | | |  __/\__ \ |_ 
 *   |_____/ \___|\___|_|___/_|\___/|_| |_|   |_| \___/|_|  \___||___/\__|
 *                                          _ _      _ 
 *                                         | | |    | |
 *                    _ __   __ _ _ __ __ _| | | ___| |
 *                   | '_ \ / _` | '__/ _` | | |/ _ \ |
 *                   | |_) | (_| | | | (_| | | |  __/ |
 *                   | .__/ \__,_|_|  \__,_|_|_|\___|_|
 *                   | |                               
 *                   |_|                               
 *  _                 _                           _        _   _             
 * (_)               | |                         | |      | | (_)            
 *  _ _ __ ___  _ __ | | ___ _ __ ___   ___ _ __ | |_ __ _| |_ _  ___  _ __  
 * | | '_ ` _ \| '_ \| |/ _ \ '_ ` _ \ / _ \ '_ \| __/ _` | __| |/ _ \| '_ \ 
 * | | | | | | | |_) | |  __/ | | | | |  __/ | | | || (_| | |_| | (_) | | | |
 * |_|_| |_| |_| .__/|_|\___|_| |_| |_|\___|_| |_|\__\__,_|\__|_|\___/|_| |_|
 *             | |                                                           
 *             |_| 
 *
 * Decision forest parallel implementation. 
 *      Extremly simple and minimalistic library.
 * 
 * version 1.0
 * 
 * The MIT License (MIT)
 * 
 * Copyright (c) 2012 Timo Petmanson
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef _DFPAR_H_
#define _DFPAR_H_

// if your compiler misses some C++11 features, you can substitute them with boost functionality
// and uncomment some of the following lines
//#define DFPAR_USE_BOOST_THREADS
//#define DFPAR_USE_BOOST_RANDOM

// strings & streams
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
// data structures
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <utility>
// math
#include <cmath>
#ifdef DFPAR_USE_BOOST_RANDOM
    #include <boost/random/mersenne_twister.hpp>
    #include <boost/random/uniform_int_distribution.hpp>
#else
    #include <random>
#endif
// other
#include <memory>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <numeric>

// in case we want to use boost threads instead of standard ones
#ifdef DFPAR_USE_BOOST_THREADS
    #include <boost/thread/thread.hpp>
    #include <boost/ref.hpp>
    typedef boost::thread dfparthread;
    #define dfparref boost::ref
#else
    #include <thread>
    typedef std::thread dfparthread;
    #define dfparref std::ref
#endif

#define DFPAR_VERSION   "1.0"

// define dfpar namespace in a more convenient way
#ifndef DFPAR_BEGIN_NAMESPACE
#define DFPAR_BEGIN_NAMESPACE namespace dfpar {
#define DFPAR_END_NAMESPACE }
#endif

DFPAR_BEGIN_NAMESPACE

// declare templates.
template<typename T, typename L> class DNode;
template<typename T, typename L> class DTree;
template<typename T, typename L> class DForest;
template<typename T, typename L> class DataSet;
template<typename T, typename L> class DTreeBuilder;
template<typename T, typename L> class DForestBuilder;
template<typename L> class ResultRow;
template<typename T> std::vector<T> sample(std::vector<T>, size_t const k);
template<typename T> std::vector<T> createRange(T begin, T end);

// declare functions
double entropy(std::vector<size_t> const&, size_t total=0);
double gain(std::vector<size_t> const&, std::vector<size_t> const&);

/////////////////////////////////////////////////////////////////////////////
// utility
/////////////////////////////////////////////////////////////////////////////

/** Function for determining, if a type is discrete.
 * By default, each type is reported as discrete, except float, double and 
 * long double.
 */
template<typename T> inline bool isDiscrete()              { return true; }
template<>           inline bool isDiscrete<float>()       { return false; }
template<>           inline bool isDiscrete<double>()      { return false; }
template<>           inline bool isDiscrete<long double>() { return false; }

// templates for default values to pass some Valgrind checks for unitialized values
// in the libraries.
template<typename T> inline T defaultValue() { return T(); }
template<>           inline int defaultValue<int>() { return 0; }
template<>           inline unsigned int defaultValue<unsigned int>() { return 0; }
template<>           inline float defaultValue<float>() { return 0; }
template<>           inline double defaultValue<double>() { return 0; }
template<>           inline long double defaultValue<long double>() { return 0; }
template<>           inline std::string defaultValue<std::string>() { return ""; }

// template stuff for comparing discrete and continuous types.
template<typename T> inline bool discreteEqual(T const& x1, T const& x2)
{
    assert(isDiscrete<T>());
    return x1 == x2;
}

template<typename T> inline bool continuousEqual(T const& x1, T const& x2)
{
    assert(!isDiscrete<T>());
    T const eps = std::numeric_limits<T>::epsilon();
    return x1 - x2 < eps && x2 - x1 < eps; 
}

template<typename T> inline bool equal(T const& x1, T const& x2)
{
    return discreteEqual(x1, x2);
}

/** Function for comparing if values x1 and x2 of type T are (almost) equal.
 * \tparam T Type of the values to compare.
 * \param x1 The first value.
 * \param x2 The second value.
 * \return true if values are (almost) equal and false otherwise.
 */
template<> inline bool equal<float>(float const& x1, float const& x2)
{
    return continuousEqual<float>(x1, x2);
}

template<> inline bool equal<double>(double const& x1, double const& x2)
{
    return continuousEqual<double>(x1, x2);
}

template<>
inline bool equal<long double>(long double const& x1, long double const& x2) 
{
    return continuousEqual<long double>(x1, x2);
}

/** Calculate the entropy.
 * \param counts The vector of class counts.
 * \param total Optional sum of the counts, if not given, the function
 *              will calculate it itself.
 * \return Entropy.
 */
inline double entropy(std::vector<size_t> const& counts, size_t total)
{
    if (total == 0) { // if counts total was not given
        total = std::accumulate(counts.begin(), counts.end(), 0);
    }
    double entropy = 0.0;
    double const epsilon = std::numeric_limits<double>::epsilon();
    for (size_t idx=0 ; idx<counts.size() ; ++idx) {
        double prob = static_cast<double>(counts[idx]) / total + epsilon;
        entropy -= prob*std::log(prob);
    }
    return entropy;
}

inline size_t gainHelper(size_t count, size_t total) {
    return total - count;
}

/** Calculate information gain of splitting the the classes described
 * by totals into two, with given respective counts of the first split.
 * \param counts The counts of classes in the first split.
 * \param totals The total number of counts of classes (in the same order)
 *               in both splits.
 * \return The information gain by splitting the samples into two.
 */
inline double gain(std::vector<size_t> const& counts, 
            std::vector<size_t> const& totals)
{
    assert (counts.size() == totals.size());
    // create the counts for the second split
    std::vector<size_t> rest(counts.size());
    std::transform(counts.begin(), counts.end(), totals.begin(), rest.begin(), gainHelper);
    // sum the totals of both splits separately
    size_t countsSum = std::accumulate(counts.begin(), counts.end(), 0);
    size_t restSum   = std::accumulate(rest.begin(), rest.end(), 0);
    // sum the elements in both splits
    size_t totalSum  = countsSum + restSum;
    
    // calculate entropies of both splits
    double H1 = entropy(counts, countsSum);
    double H2 = entropy(rest, restSum);
    // calculate the information gain
    double sum = static_cast<double>(countsSum) / totalSum * H1 +
                 static_cast<double>(restSum)   / totalSum * H2;
    return entropy(totals) - sum;
}

/** Function for taking a random sample without replacement.
 * \param values The vector of values describing the population.
 * \param k The sample size.
 * \tparam T the type of the vector elements.
 * \return Vector containing a sample from the population.
 */
template<typename T>
std::vector<T> sample(std::vector<T> values, size_t const k)
{
#ifdef DFPAR_USE_BOOST_RANDOM
    static boost::random::mt19937 rng;    // the randomness sources
#else
    static std::mt19937 rng;    // the randomness sources
#endif
    
    // check if it is possible to take a sample without replacement
    if (k > values.size()) {
        std::stringstream ss;
        ss << "Size of requested sample" << k;
        ss << " is larger than size of given vector " << values.size();
        throw std::runtime_error(ss.str());
    }
    // create the sample
    std::vector<T> sample;
    sample.reserve(k);
    for (size_t idx=0 ; idx<k ; ++idx) {
#ifdef DFPAR_USE_BOOST_RANDOM
        boost::random::uniform_int_distribution<size_t> dist(0, values.size() - idx - 1);
#else
        std::uniform_int_distribution<size_t> dist(0, values.size() - idx - 1);
#endif
        size_t elemIdx = values[dist(rng)];
        sample.push_back(values[elemIdx]);
        values[elemIdx] = values[values.size() - idx - 1];
    }
    return sample;
}

/** Create a range [begin, end) of type T. the operator++ is used to increment
 * from the begin value greater or equal to end is reached.
 * \param begin The first value of the range.
 * \param end The first value excluded from the range.
 * \return The vector<T> instance containing the range.
 */
template<typename T>
std::vector<T> createRange(T begin, T end)
{
    std::vector<T> range;
    range.reserve(end - begin);
    for (T value=begin ; value<end ; ++value) {
        range.push_back(value);
    }
    return range;
}

/////////////////////////////////////////////////////////////////////////////
// decision forest structure
/////////////////////////////////////////////////////////////////////////////

/** Decision forest class.
 * \tparam T Typename for feature data.
 * \tparam L Typename for response (label) data.
 */
template<typename T, typename L>
class DForest
{
protected:
    /// Internal trees of the forest.
    std::vector<DTree<T, L>> trees;
    /// Estimated out-of-bag error for the forest.
    double oobError;
    
    /** Method combining individual tree classifications into
     * predicted labels and probabilities.
     * \param ds The dataset.
     * \param responses vector of tree responses for all dataset rows.
     */
    std::vector<ResultRow<L>> combineResponses(
        DataSet<T, L> const& ds, 
        std::vector<std::vector<L>> const& responses)
    {
        assert (trees.size() == responses.size());
        std::vector<ResultRow<L>> rows;
        rows.reserve(ds.numRows());
        
        for (size_t row=0 ; row<ds.numRows() ; ++row) {
            std::map<L, double> counts;
            for (size_t idx=0 ; idx<responses.size() ; ++idx) {
                counts[responses[idx][row]] += 1.0;
            }
            // count the probabilities
            double bestprob = std::numeric_limits<double>::min();
            L response = ds.getY(0);
            for (auto i=counts.begin() ; i != counts.end() ; ++i) {
                i->second /= responses.size();
                if (i->second > bestprob) {
                    response = i->first;
                    bestprob = i->second;
                }
            }
            // assmeble the ResultRow instance.
            ResultRow<L> resrow(response, counts);
            rows.push_back(resrow);
        }
        
        return rows;
    }
    
    /** Update the decision forest oob error estimate. */
    void updateOobError()
    {
        oobError = 0;
        for (size_t treeIdx=0 ; treeIdx < trees.size() ; ++treeIdx) { 
            DTree<T,L> const& tree = trees[treeIdx];
            oobError += tree.getOobError();
        }
        oobError /= trees.size();
    }
    
public:
    /** Initiate an empty decision forest instance. */
    DForest()
    { }
    
    /** Initiate a decision forest instance from given list of trees. */
    DForest(std::vector<DTree<T, L>> const& trees) : trees(trees)
    {
        updateOobError();
    }
    
    /** Merge the trees of otherForest to this tree. */
    void merge(DForest<T,L> const& otherForest) {
        trees.insert(trees.end(), otherForest.trees.begin(), 
                                  otherForest.trees.end());
        updateOobError();
    }
    
    /** Classify all rows in the dataset.
     * \param ds The dataset.
     * \param numThreads How many threads to use for dividing up the work.
     */
    std::vector<ResultRow<L>> classifyAll(DataSet<T, L> const& ds, 
                                          size_t const numThreads)
    {
        std::vector<std::vector<L>> responses(trees.size());
        std::vector<dfparthread> threads;
        std::vector<size_t> treeIndices;
        
        for (size_t treeIdx=0 ; treeIdx<trees.size() ; ++treeIdx) {
            while (threads.size() >= numThreads) {
                for (size_t thIdx=0 ; thIdx<threads.size() ; ++thIdx) {
                    if (responses[treeIndices[thIdx]].size() == ds.numRows()) {
                        threads[thIdx].join();
                        threads.erase(threads.begin() + thIdx);
                        treeIndices.erase(treeIndices.begin() + thIdx);
                    }
                }
            }
            
            // add thread
            responses[treeIdx].resize(ds.numRows());
            threads.push_back(dfparthread(
                &DTree<T, L>::classifyAll,
                dfparref(trees[treeIdx]),
                dfparref(ds),
                dfparref(responses[treeIdx])));
            treeIndices.push_back(treeIdx);
            
            trees[treeIdx].classifyAll(ds, responses[treeIdx]);
        }
        // wait the rest of the threads
        while (threads.size() > 0) {
            threads[0].join();
            threads.erase(threads.begin());
        }
        
        return combineResponses(ds, responses);
    }
    
    /** Get the out-of-bag error estimate for the decision forest. */
    double getOobError() const
    {
        return oobError;
    }
    
    // serialization
    friend std::ostream& operator<<(std::ostream& os, DForest<T, L> &forest)
    {
        os << std::string(DFPAR_VERSION) << std::endl;
        os << forest.trees.size() << std::endl;
        for (size_t treeIdx=0 ; treeIdx < forest.trees.size() ; ++treeIdx) {
            DTree<T,L> &tree = forest.trees[treeIdx];
            os << tree;
        }
        return os;
    }
    
    // deserialization
    friend std::istream& operator>>(std::istream& is, DForest<T, L> &forest)
    {
        // parse version, may be necessary, if format changes in the future.
        std::string version;
        std::getline(is, version);
        // parse number of trees
        std::string line;
        std::getline(is, line);
        size_t nTrees = std::stoul(line);
        // clear current trees (if any) and resize the vector.
        forest.trees.clear();
        forest.trees.resize(nTrees);
        // read individual trees.
        for (size_t treeIdx=0 ; treeIdx < forest.trees.size() ; ++treeIdx) {
            DTree<T,L> &tree = forest.trees[treeIdx];
            is >> tree;
        }
        forest.updateOobError();
        return is;
    }
    
    friend class DForestBuilder<T, L>;
};

/**
 * DTree is a component of decision forest.
 * \tparam T Typename for feature data.
 * \tparam L Typename for response (label) data.
 */
template<typename T, typename L>
class DTree
{
protected:
    /// Shared pointer to the root node.
    std::shared_ptr<DNode<T, L>> root;
    /// oobError calculated in trainign phase.
    double oobError;
public:
    DTree()
    {
        oobError = 1;
    }
    
    L classify(DataSet<T, L> const& ds, size_t const idx)
    {
        assert (idx < ds.numRows());
        return root->classify(ds, idx);
    }
    
    // result, must be already initialized vector.
    void classifyAll(DataSet<T, L> const& ds, std::vector<L> &result)
    {
        for (size_t idx=0 ; idx<ds.numRows() ; ++idx) {
            result[idx] = classify(ds, idx);
        }
    }
    
    std::shared_ptr<DNode<T, L>> getRoot()
    {
        return root;
    }
    
    double getOobError() const {
        return oobError;
    }
    
    friend std::ostream& operator<<(std::ostream& os, DTree<T, L> &tree)
    {
        os << tree.oobError << std::endl;
        os << (*tree.root);
        return os;
    }
    
    friend std::istream& operator>>(std::istream& is, DTree<T, L> &tree)
    {
        std::string line;
        getline(is, line);
        tree.oobError = std::stod(line);
        tree.root = std::shared_ptr<DNode<T,L>>(new DNode<T,L>);
        is >> (*tree.root);
        
        return is;
    }
    
    friend class DTreeBuilder<T, L>;
};

/** A single node in a decision tree.
 * \tparam T Typename for feature data.
 * \tparam L Typename for response (label) data.
 */
template<typename T, typename L>
class DNode
{
protected:
    /// The cutoff value of the branch. Tested as x >= cutoff.
    T cutoff;
    /// If the node is a leaf node, then the label assigned to it.
    L label;
    /// Feature (column) index the split is made.
    size_t featIdx;
    
    /// Left child node.
    std::shared_ptr<DNode<T, L>> left;
    /// Right child node.
    std::shared_ptr<DNode<T, L>> right;
public:
    DNode()
    {
        cutoff  = 0;
        label   = defaultValue<L>();
        featIdx = 0;
    }
    
    /** \return true if node is leaf, false otherwise. */
    bool isLeaf()
    {
        return left == NULL && right == NULL;
    }
    
    /** Classify a row in dataset.
     * \param ds Dataset instance.
     * \param idx The row index.
     */
    L classify(DataSet<T, L> const& ds, size_t const idx)
    {
        if (isLeaf()) {
            return label;
        } else {
            if (ds.get(idx, featIdx) >= cutoff) {
                return left->classify(ds, idx);
            } else {
                return right->classify(ds, idx);
            }
        }
    }
    
    // serialization.
    friend std::ostream& operator<<(std::ostream& os, DNode<T, L> &node)
    {
        os << node.cutoff << std::endl << node.label;
        os << std::endl << node.featIdx << std::endl;
        if (node.left == NULL) {
            os << std::endl;
        } else {
            os << "LT" << std::endl << (*node.left);
        }
        if (node.right == NULL) {
            os << std::endl;
        } else {
            os << "RT" << std::endl << (*node.right);
        }
        return os;
    }
    
    // deserialization.
    friend std::istream& operator>>(std::istream& is, DNode<T, L> &node)
    {
        std::string line;
        std::stringstream ss;
        // convert cutoff value
        getline(is, line);
        ss << line;
        ss >> node.cutoff;
        // convert response value
        getline(is, line);
        ss.clear();
        ss << line;
        ss >> node.label;
        // convert feature index of split
        getline(is, line);
        node.featIdx = std::stoull(line);
        // try to parse left branch
        getline(is, line);
        if (line.size() > 1) {
            node.left = std::shared_ptr<DNode<T,L>>(new DNode<T, L>);
            is >> (*node.left);
        }
        // try to parse right branch
        getline(is, line);
        if (line.size() > 1) {
            node.right = std::shared_ptr<DNode<T,L>>(new DNode<T, L>);
            is >> (*node.right);
        }
        return is;
    }
    
    friend class DTreeBuilder<T, L>;
};

/////////////////////////////////////////////////////////////////////////////
// data
/////////////////////////////////////////////////////////////////////////////

/** Dataset class used by RFPAR library.
 * \tparam T Typename for feature data.
 * \tparam L Typename for response (label) data.
 */
template<typename T, typename L>
class DataSet
{
    size_t N, M;    // Number of rows and columns in dataset.
    T* X;           // Underlying feature data matrix.
    L* y;           // Underlaying response value (label) vector.
    bool ownsData;  // Does the DataSet instance own underlying X and y data?
    
protected:
    /** Create an empty dataset. No memory is acquired for underlying data. */
    void makeEmpty()
    {
        N = M = ownsData = 0;
        X = NULL; y = NULL;
    }
    
    /** Create a dataset of size N*M.
     * Method acquires memory for feature matrix as well as for response matrix.
     * The values are left uninitialized, unless typenames T and L have default
     * constructors, which is not the case for ordinary int, float and double
     * types.
     * 
     * \param N the number of rows in the initialized dataset.
     * \param M the number of columns in the initialized dataset.
     */
    void makeUninitialized(size_t const N, size_t const M)
    {
        this->N = N, this->M = M;
        ownsData = true;
        X = new T[N * M];
        y = new L[N];
    }
    
    /** Free the memory used by this DataSet instance, given that
     * the instance owns the memory.
     */
    void destruct()
    {
        if (ownsData) {
            delete[] X;
            delete[] y;
        }
    }
    
public:
    /** Construct an empty dataset. No memory is acquired for underlying data.*/
    DataSet()
    {
        makeEmpty();
    }
    
    /** Create a dataset of size N*M.
     * Method acquires memory for feature matrix as well as for response matrix.
     * The values are left uninitialized, unless typenames T and L have default
     * constructors, which is not the case for ordinary int, float and double
     * types.
     * 
     * \param N the number of rows in the initialized dataset.
     * \param M the number of columns in the initialized dataset.
     */
    DataSet(size_t const N, size_t const M)
    {
        makeUninitialized(N, M);
    }

    /** Create a dataset of size N*M.
     * 
     * \param N the number of rows in the initialized dataset.
     * \param M the number of columns in the initialized dataset.
     * \param X Pointer to feature matrix data.
     * \param y Pointer to response vector data.
     * \param ownsData If true, then releases the pointer to feature and
     *                 response data.
     */
    DataSet(size_t const N, size_t const M, 
            T const* X, L const* y, bool ownsData=false)
        : N(N), M(M), X(X), y(y), ownsData(ownsData)
    {
    }

    /** Destructor. 
     * If ownsData is set true, releases memory for feature and response data.
     */
    ~DataSet()
    {
        destruct();
    }

    /** \return The number of rows in the dataset. */
    size_t numRows() const
    {
        return N;
    }

    /** \return The number of columns in the dataset. */
    size_t numCols() const
    {
        return M;
    }

    /** Set the matrix element at row and column to given value. 
     * \param row Given row index.
     * \param col Given column index.
     * \param The value to assign to the feature matrix cell at (row, col).
     */
    void set(size_t const row, size_t const col, T const value)
    {
        X[row*M + col] = value;
    }

    /** Get the copy of the value at given row and col of the matrix.
     * \param row The given row index.
     * \param col The given column index.
     * \return The copy of value at given row and column in feature matrix.
     */
    double get(size_t const row, size_t const col) const
    {
        return X[row*M + col];
    }

    /** Set the response value for given row.
     * \param row The given row index.
     * \param value The response value to assign to given row.
     */
    void setY(size_t const row, L const& value)
    {
        y[row] = value;
    }
    
    /** Get the response value assigned to given row index.
     * \param row The given row index.
     * \return The assigned response value.
     */
    L getY(size_t const row) const
    {
        return y[row];
    }
    
    /** Get the index of the element in data for given row and col indices.
     * \param row The given row index.
     * \param col The given col index.
     * \return The element index in internal data pointer.
     */
    size_t getIndex(size_t const row, size_t const col) const 
    {
        return row*M + col;
    }


    /** 
     * Get the pointer to internal feature matrix of size N*M.
     * Data is stored in a single array of size N*M with following structure:
     * row_1: elem_1, elem_2, elem_3, ..., elem_M
     * row_2: elem_{M+1}, ..,              elem{2M}
     *       ...
     * row_N: ...                          elem{NM}
     * 
     * that is, it is stored row by row, starting from the first. Note that the
     * above example used 1-based inidices, whereas in code you 
     * use 0-based indices.
     * 
     * \return The pointer to internal feature matrix.
     */
    T* getFeatureMatrix() const
    {
        return X; 
    }

    /** 
     * Replace the internal data array of the dataset.
     * 
     * The caller must ensure that data represents N rows and M columns. Also, 
     * if this DataSet instance is the owner of the data, it tries to free the 
     * memory of that pointer upon destruction.
     * \param data The pointer to new data. Data is not copied!
     */
    void setFeatureMatrix(const T* X)
    {
        X = X;
    }
    
    /** \return Pointer to underlying response vector. */
    L* getResponseVector() const
    {
        return y;
    }
    
    /** Replace the internal response vector of the dataset. Also, if this
     * DataSet instance is the owner of the data, it tries to free the memory of
     * that spointer upon destruction.
     * \param y The pointer to assign. Response values are not copied!
     */
    void setResponseVector(const L* y)
    {
        y = y;
    }
    
    /** \return true, if the DataSet instance is responsible for releasing the 
     * memory upon destruction. */
    bool getOwnsData() const
    {
        return ownsData;
    }
    
    /** Tell the DataSet instance, if it is responsible for releasing the memory 
     * upon destruction. */
    void setOwnsData(bool const ownsData)
    {
        this->ownsData = ownsData;
    }
    
    /** Method from creating random test and train datasets.
     * \param test Uninitialized reference for train dataset.
     * \param train Uninitialised reference for test dataset.
     * \param proportion Value in range (0,1) determining how large proportion should be in trainset.
     */
    void createTrainAndTest(DataSet<T, L>& train, DataSet<T, L>& test, double proportion=0.8) {
        if (!(proportion > 0 && proportion < 1)) {
            throw new std::runtime_error("Proportion must be value in range (0,1)!");
        }
        if (numRows() < 2) {
            throw new std::runtime_error("The dataset has less than two rows!");
        }
        // decide the number of example in the training set
        size_t numInTrain = static_cast<size_t>(static_cast<double>(numRows()) * proportion);
        if (numInTrain == 0) {
            numInTrain = 1;
        } else if (numInTrain == numRows()) {
            numInTrain -= 1;
        }
        // take a random sample of examples
        auto indices = createRange<size_t>(0, numRows()-1);
        indices = sample(indices, numInTrain);
        
        // set up train and test datasets
        train.makeUninitialized(numInTrain, numCols());
        test.makeUninitialized(numRows() - numInTrain, numCols());
        // divide the data into train and test datasets
        std::sort(indices.begin(), indices.end());
        {
            size_t idx1 = 0; // index for train dataset
            size_t idx2 = 0; // index for test dataset
            size_t origIdx = 0; // index in original dataset
            for (size_t nextTrain=0 ; nextTrain < indices.size() ; ++nextTrain) {
                while (origIdx < indices[nextTrain]) {
                    // add to test set
                    size_t thisStart = getIndex(origIdx, 0);
                    size_t testStart = test.getIndex(idx2, 0);
                    memmove(&test.X[testStart], &this->X[thisStart], numCols()*sizeof(T));
                    test.setY(idx2, this->getY(origIdx));
                    
                    ++idx2;
                    ++origIdx;
                }
                // add next to train set
                size_t thisStart = getIndex(origIdx, 0);
                size_t trainStart = train.getIndex(idx1, 0);
                memmove(&train.X[trainStart], &this->X[thisStart], numCols()*sizeof(T));
                train.setY(idx1, this->getY(origIdx));
                
                ++origIdx;
                ++idx1;
            }
            // append rest to test set
            while (origIdx < numRows()) {
                // add to test set
                size_t thisStart = getIndex(origIdx, 0);
                size_t testStart = test.getIndex(idx2, 0);
                memmove(&test.X[testStart], &this->X[thisStart], numCols()*sizeof(T));
                test.setY(idx2, this->getY(origIdx));
                
                ++idx2;
                ++origIdx;
            }
        }
    }
    
    /** Dataset serialization operator.
     * The user must know the type of features and response values.
     * \param os The output stream.
     * \param dataset The dataset.
     * \return The given output stream os.
     */
    friend std::ostream& operator<<(std::ostream& os, DataSet<T,L> const& ds)
    {
        size_t numElems = ds.N*ds.M;
        os << ds.N << " " << ds.M << std::endl;
        for (size_t elem=0 ; elem<numElems ; ++elem) {
            // should we separate it from last element by a space?
            if (elem % ds.M > 0) { 
                os << " ";
            } 
            os << ds.X[elem];
            // should we print the label here?
            if (elem % ds.M == ds.M - 1) { 
                os << " " << ds.y[elem/ds.M] << std::endl;
            }
        }
        return os;
    }
    
    /** Dataset deserialization operator.
     * The user must know the type of features and response values.
     * \param is The input stream.
     * \param dataset The dataset.
     * \return The given input stream is.
     */
    friend std::istream& operator>>(std::istream& is, DataSet<T,L>& ds)
    {
        size_t N, M;
        is >> N >> M;
        
        size_t numElems = N*M;
        ds.destruct();
        ds.makeUninitialized(N, M);
        
        for (size_t elem=0 ; elem<numElems ; ++elem) {
            is >> ds.X[elem];
            if (elem % M == M - 1) { // read the label
                is >> ds.y[elem/M];
            }
        }
        return is;
    }
};

/** ResultRow describes the preferred response value (label) with probabilities
 * of all candidates considered for classification.
 * \tparam L Typename for response (label) data.
 */
template<typename L>
struct ResultRow
{
protected:
    /// Response value.
    L response;
    /// Map of response values and their probabilities.
    std::map<L, double> probs;
    
public:
    /** Get the copy of response (label) of this row. */
    L getResponse() const
    {
        return response;
    }
    
    /** Return the probability for response value (label) y.
     * \param y The response value (label).
     * \return the probability for label y.
     */
    double getProbability(L const& y)
    {
        auto iter = probs.find(y);
        if (iter != probs.end()) {
            return iter->second;
        }
        return 0;
    }

    /** Is the resultrow initialized.
     * \return true or false.
     */
    bool isEmpty() const
    {
        return probs.size() > 0;
    }
    
    /** Construct a Resultrow instance from given data */
    ResultRow(L response, std::map<L, double> const& probs) :
        response(response), probs(probs)
    {       
    }
};

/////////////////////////////////////////////////////////////////////////////
// builders
/////////////////////////////////////////////////////////////////////////////

/** Decision Tree builder class.
 * \tparam T Typename for feature data.
 * \tparam L Typename for response (label) data.
 */
template<typename T, typename L>
class DTreeBuilder
{
    /// Reference to the dataset used to build the tree.
    DataSet<T, L> const& ds;
    /// Number of rows to sample when building the tree.
    size_t const size;
    /// Number of features to sample.
    size_t const numFeat;
    /// Sample indices for building this tree.
    std::vector<size_t> idxs;
    
    /** Test if the given feature is constant in given sample indices. */
    bool responseValuesSame(std::vector<size_t> const& idxs) const
    {
        assert (idxs.size() > 0);
        L y = ds.getY(idxs[0]);
        for (size_t i=0 ; i < idxs.size() ; ++i) {
            size_t idx = idxs[i];
            if (!equal<L>(y, ds.getY(idx))) {
                return false;
            }
        }
        return true;
    }
    
    /** Test if the given feature is constant in given sample indices. */
    bool isConstant(std::vector<size_t> const& idxs, size_t const featIdx) const
    {
        assert (idxs.size() > 0);
        T x = ds.get(idxs[0], featIdx);
        for (size_t i=0 ; i < idxs.size() ; ++i) {
            size_t idx = idxs[i];
            if (!equal<T>(ds.get(idx, featIdx), x)) {
                return false;
            }
        }
        return true;
    }
    
    /** Get the most frequent response value in given sample rows. */
    L modeResponse(std::vector<size_t> const& idxs) const
    {
        assert (isDiscrete<L>());
        assert (idxs.size() > 0);
        
        std::unordered_map<L, size_t> counts;
        size_t bestCount = 0;
        L bestY = ds.getY(0);
        for (size_t i=0 ; i < idxs.size() ; ++i) {
            size_t idx = idxs[i];
            L y = ds.getY(idx);
            size_t count = counts[y] + 1;
            counts[y] = count;
            if (count > bestCount) {
                bestCount = count;
                bestY = y;
            }
        }
        return bestY;
    }
    
    std::pair<T, double> split(std::vector<size_t> const& idxs,
                               size_t const featIdx)
    {
        assert (isDiscrete<L>());
        assert (idxs.size() > 1);
        
        // Get the values and responses for given sample in given column.
        // Additionally count how many responses (class labels) there are.
        std::vector<std::pair<T, L> > values;
        std::unordered_map<L, size_t> totalCounts;
        for (size_t i=0 ; i < idxs.size() ; ++i) {
            size_t idx = idxs[i];
            L y = ds.getY(idx);
            totalCounts[y] = totalCounts[y] + 1;
            values.push_back(std::pair<T, L>(ds.get(idx, featIdx), y));
        }
        
        // sort them by value, so we can efficiently determine the cutoff value
        std::sort(values.begin(), values.end());
        
        // determine the best cutoff point
        std::unordered_map<L, size_t> counts;
        double bestGain = std::numeric_limits<double>::min();
        T bestValue = (values.end()-1)->first;
        
        for (auto it = values.begin() ; it != values.end() ; ++it) {
            // we can calculate the information gain on points, where value is
            // being splitted if it is the last value
            if (it != values.begin() && !equal<T>(it->first, (it-1)->first)) {
                // get the counts as a vector
                std::vector<size_t> countVec;
                std::vector<size_t> totalCountVec;
                for (auto iter = totalCounts.begin() ; iter != totalCounts.end() ; ++iter) {
                    countVec.push_back(counts[iter->first]);
                    totalCountVec.push_back(iter->second);
                }
                double splitgain = gain(countVec, totalCountVec);
                if (splitgain > bestGain) {
                    bestGain  = splitgain;
                    if (isDiscrete<T>()) { // we cannot always take reasonable avereage of discrete values
                        bestValue = it->first;
                    } else { // but we can do so with continuous values.
                        bestValue = (it->first + (it-1)->first) / 2;
                    }
                }
            }
            // update the counts of positive and negative labels when splitting
            // at iter->first
            counts[it->second] = counts[it->second] + 1;
        }
        return std::pair<T, double>(bestValue, bestGain);
    }
    
    /** Given an empty node, choose a sample of features and splitting point 
     * for it.
     * \param node The empty DNode instance that is filled with information.
     * \param sampleIndices The rows of dataset used for determining splitting
     *                      point of this node.
     * \param return true, if the initiated node is leaf node.
     */
    bool findAndSetBestSplit(DNode<T, L>& node,
                             std::vector<size_t> const& idxs)
    {
        assert (idxs.size() > 0);
        if (responseValuesSame(idxs)) {
            node.label = ds.getY(idxs[0]);
            return true;
        }
        
        double bestGain = std::numeric_limits<double>::min();
        std::vector<size_t> featureIndices =
            sample(createRange<size_t>(0, ds.numCols()), numFeat);
        
        for (size_t i = 0 ; i < featureIndices.size() ; ++i) {
            size_t featIdx = featureIndices[i];
            // if current feature on sample is constant
            if (isConstant(idxs, featIdx)) {
                // and we have not found anything good yet.
                if (bestGain == std::numeric_limits<double>::min()) {
                    // then assign the most frequent label to it.
                    node.cutoff = ds.get(idxs[0], featIdx);
                    node.label  = modeResponse(idxs);
                }
            } else {
                std::pair<T, double> best = split(idxs, featIdx);
                if (best.second > bestGain) {
                    node.cutoff     = best.first;
                    bestGain        = best.second;
                    node.featIdx    = featIdx;
                }
            }
        }
        // we have not found anything very good, this is the leaf node.
        if (bestGain == std::numeric_limits<double>::min()) {
            node.label  = modeResponse(idxs);
            return true;
        }
        return false;
    }
    
    /** Create sample indices for children of the given node.
     * \param node The parent node.
     * \param sampleIndices The training indices of the parent node.
     * \param partitionA The empty vector to store the indices of first child.
     * \param partitionB The empty vector to store the indices of the second 
     *                   child.
     */
    void createChildNodePartitions(
        DNode<T, L> const& node, 
        std::vector<size_t> const& idxs, 
        std::vector<size_t>& partitionA, 
        std::vector<size_t>& partitionB) 
    {
        for (size_t idx=0 ; idx<idxs.size() ; ++idx) {
            size_t row = idxs[idx];
            if (ds.get(row, node.featIdx) >= node.cutoff) {
                partitionA.push_back(row);
            } else {
                partitionB.push_back(row);
            }
        }
    }
    
    std::shared_ptr<DNode<T, L>> _build(std::vector<size_t> idxs)
    {
        std::shared_ptr<DNode<T, L>> node(new DNode<T, L>);
        bool isLeaf = findAndSetBestSplit(*node, idxs);

        if (!isLeaf) {
            std::vector<size_t> partitionA;
            std::vector<size_t> partitionB;
            createChildNodePartitions(*node, idxs, partitionA, partitionB);

            node->left  = _build(partitionA);
            node->right = _build(partitionB);
        }
        
        return node;
    }
    
    std::vector<size_t> getTestIndices()
    {
        std::sort(idxs.begin(), idxs.end());
        std::vector<size_t> testIdxs;
        size_t lastIdx = 0;
        for (size_t i=0 ; i < idxs.size() ; ++i) {
            size_t idx = idxs[i];
            while (lastIdx < idx) {
                testIdxs.push_back(lastIdx);
                lastIdx += 1;
            }
            lastIdx += 1;
        }
        while (lastIdx < ds.numRows()) {
            testIdxs.push_back(lastIdx++);
        }
        return testIdxs;
    }
    
public:
    /** Construct a new decision tree builder.
     * \param ds The dataset used for training.
     * \param size The size of the bootstrap sample to use.
     * \param numFeat The number of random features to use.
     */
    DTreeBuilder(DataSet<T, L> const& ds, 
                 size_t const size, 
                 size_t const numFeat)
            : ds(ds), size(size), numFeat(numFeat) 
    {
        if (size <= 1) {
            throw std::runtime_error("Sample size must be larger than 1.");
        }
        if (numFeat < 1) {
            std::string err = "Number of sample features must be at least 1.";
            throw std::runtime_error(err);
        }
        if (size >= ds.numRows()) {
            std::string err = "Sample size must be less than number of rows "
                " in dataset.";
            throw std::runtime_error(err);
        }
        if (numFeat > ds.numCols()) {
            std::string err = "Number of features must be less than number of"
                " columns in dataset.";
            throw std::runtime_error(err);
        }
        idxs = sample(createRange<size_t>(0, ds.numRows()), size);
    }
    
    /** Train a given tree instance. */
    void build(DTree<T, L>& tree)
    {
        tree.root = _build(idxs);
        
        // calculate the OOB error for this tree.
        size_t incorrect = 0;
        std::vector<size_t> testIdxs = getTestIndices();
        for (size_t i=0 ; i < testIdxs.size() ; ++i) {
            size_t idx = idxs[i];
            L response = tree.classify(ds, idx);
            incorrect += response != ds.getY(idx);
        }
        tree.oobError = static_cast<double>(incorrect) / testIdxs.size();
    }
};

/** Class for training decision forests.
 * \tparam T Typename for feature data.
 * \tparam L Typename for response (label) data.
 */
template<typename T, typename L>
class DForestBuilder
{
protected:
    DataSet<T, L> const& ds;    // dataset used for training.
    size_t bootStrapSize;
    size_t numTrees;
    size_t numRandomFeatures;
    
    void buildSingleTree(DTree<T, L>& tree)
    {
        try {
            DTreeBuilder<T, L> builder(ds, bootStrapSize, numRandomFeatures);
            builder.build(tree);
        } catch (std::exception &e) {
            std::cerr << e.what() << std::endl;
        } catch (const char* e) {
            std::cerr << e << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception!" << std::endl;
        }
    }
    
public:
    /** Initialize a decision forest builder.
     * \param ds The dataset to use for training.
     */
    DForestBuilder(DataSet<T,L> const& ds) : ds(ds)
    {
        if (ds.numRows() < 3) {
            throw std::runtime_error("Dataset must have at least 3 rows!");
        }
        if (ds.numCols() == 0) {
            throw std::runtime_error("Dataset must have at least one feature!");
        }
        bootStrapSize = ds.numRows() / 3 * 2;
        numTrees = 500;
        numRandomFeatures = std::min<int>(ds.numCols(), 2);
    }
    
    /** \return The size of bootstrap sample used in training. */
    size_t getBootStrapSize() const
    {
        return bootStrapSize;
    }
    
    /** Set the size of bootstrap sample to be used in training. */
    void setBoostStrapSize(size_t const bootStrapSize)
    {
        if (bootStrapSize >= ds.numRows()) {
            throw std::runtime_error("Booststrap size must be"
                " less than number of rows in dataset!");
        }
        if (bootStrapSize == 0) {
            throw std::runtime_error("Booststrap size must be"
                " greater than 0!");
        }
        this->bootStrapSize = bootStrapSize;
    }
    
    /** \return The number of trees to train for the decision forest. */
    size_t getNumTrees() const
    {
        return numTrees;
    }
    
    /** \param The number of trees to train for the decision forest. */
    void setNumTrees(size_t const numTrees)
    {
        if (numTrees == 0) {
            throw std::runtime_error("Number of trees must be at least 1!");
        }
        this->numTrees = numTrees;
    }
    
    size_t getNumRandomFeatures() const
    {
        return numRandomFeatures();
    }
    
    void setNumRandomFeatures(size_t const numRandomFeatures)
    {
        if (numRandomFeatures == 0) {
            throw std::runtime_error("Number of random features must be"
                " at least 1");
        }
        if (numRandomFeatures > ds.numCols()) {
            throw std::runtime_error("Number of random features cannot"
                " exceed number of features in dataset!");
        }
        this->numRandomFeatures = numRandomFeatures;
    }
    
    /** Compute the decision forest.
     * \param numThreads The number of threads to use for training the model.
     */
    DForest<T, L> build(size_t numThreads=1)
    {
        std::vector<DTree<T, L>> trees(numTrees);
        std::vector<dfparthread> threads;
        std::vector<size_t> treeIndices;
         
        size_t treeIdx = 0;
        while (treeIdx < numTrees) {
            // can we join some threads.
            for (size_t idx=0 ; idx<treeIndices.size() ; ++idx) {
                // if the tree node is not not null, we can
                // join the thread.
                if (trees[treeIndices[idx]].getRoot() != NULL) {
                    threads[idx].join();
                    threads.erase(threads.begin() + idx);
                    treeIndices.erase(treeIndices.begin() + idx);
                }
            }
            // can we add new threads.
            while (threads.size() < numThreads && treeIdx < numTrees) {
                threads.push_back(dfparthread(&DForestBuilder::buildSingleTree, 
                                   *this, 
                                   dfparref(trees[treeIdx])));
                treeIndices.push_back(treeIdx);
                treeIdx++;
            }
        }
        // join the rest of the threads
        for (size_t threadIdx=0 ; threadIdx<threads.size() ; ++threadIdx) {
            threads[threadIdx].join();
        }
        return DForest<T, L>(trees);
    }
};

DFPAR_END_NAMESPACE
#endif
