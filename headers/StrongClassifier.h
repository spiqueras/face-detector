#ifndef STRONGCLASSIFIER_H
#define STRONGCLASSIFIER_H

#include "Classifier.h"
#include <algorithm>

#include <cmath>
#include <cstring>

#include "Global.h"
#include "pugixml.hpp"

class StrongClassifier{
    
    Classifier** ccls;
    FLOAT** alpha;
    FLOAT thr;
    int nCls;  // Number of weak classifiers

  public:

    StrongClassifier():thr(0), nCls(0){};

    void opt_thr(FixedImage* imgs, uint32_t nImgs, FLOAT max_fnr, bool verbose = false);

    FLOAT get_val(const Image &img);

    /** classify
    Classify an integral image
    **/
    Class classify(const Image &img);

    /** test
    Test the classifier

    **/
    void test(FixedImage* imgs, uint32_t nImgs, Class* hts = nullptr);

    /** adaBoost
    Boost the classifier

    \param imgs Array of labeled images
    \param nImgs Number of images
    \param cls Array of arrays of classifiers
    \param T number of weak classifiers
    **/
    void adaBoost(FixedImage* imgs, uint32_t nImgs, Classifier** cls, uint32_t* nCls, uint32_t NCLSTS, int T);


    /** rareBoost
    Boost the classifier

    \param imgs Array of labeled images
    \param nImgs Number of images
    \param cls Array of arrays of classifiers
    \param T number of weak classifiers
    **/
    void rareBoost(FixedImage* imgs, uint32_t nImgs, Classifier** cls, uint32_t* nCls, uint32_t NCLSTS, int T);

    void from_XML(std::string file);
    void to_XML(std::string file) const;
    
    bool from_XML_node(pugi::xml_node& pnode);
    void add_to_XML_node(pugi::xml_node& pnode) const;
    
};

#endif // STRONGCLASSIFIER_H
