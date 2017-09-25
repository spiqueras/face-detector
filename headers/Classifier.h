#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>  // sort
#include <cstring> // strcmp
#include "pugixml.hpp"

#include "Global.h"
#include "Image.h"

class Classifier{

    // struct for get_e_opt_thr
    struct Othr_aux{
      FLOAT val;
      uint32_t i;
    };

  public:

    Classifier(): thr(0){};

    virtual void set_ps(Point* ps) = 0;
    virtual void print() const= 0;

    /** classify
    Classify an integral image
    **/
    Class classify(const Image &img) const;  

    virtual FLOAT get_val(const Image &img) const= 0;

    /** get_e_opt_thr
    Find and set the optimal threshold given an array of images
    and an array of weights, and return the error value
    **/
    FLOAT get_e_opt_thr(FixedImage* imgs, uint32_t nImgs, FLOAT* D);

    virtual void add_to_XML_node(pugi::xml_node& pnode, FLOAT* alpha) const = 0;
    virtual void from_XML_node(pugi::xml_node& pnode) = 0;

  protected:

    FLOAT thr; 
    int p; // Polarity

};

// 2 zones classifier

class Z2Classifier: public Classifier{
    
    struct Points{
      Point z11; // 1st zone, up left
      Point z12; // 1st zone, down right

      Point z21; // 2nd zone, up left
      Point z22; // 2nd zone, down right
    };

    Points ps;

  public:

    Z2Classifier(){};
    Z2Classifier(Point ps[4]);
    Z2Classifier(const Z2Classifier& other);

    void set_ps(Point* ps);

    void print() const; 

    FLOAT get_val(const Image &img) const;

    void add_to_XML_node(pugi::xml_node& pnode, FLOAT* alpha) const;
    void from_XML_node(pugi::xml_node& pnode);
};

// 3 zones classifier

class Z3Classifier: public Classifier{
    
    struct Points{
      Point z11; // 1st zone, up left
      Point z12; // 1st zone, down right

      Point z21; // 2nd half, up left
      Point z22; // 2nd half, down right

      Point z31; // 3rd half, up left
      Point z32; // 3nd half, down right
    };

    Points ps;

  public:

    Z3Classifier(){};
    Z3Classifier(Point ps[4]);
    Z3Classifier(const Z3Classifier& other);

    void set_ps(Point* ps);

    void print() const;

    FLOAT get_val(const Image &img) const;

    void add_to_XML_node(pugi::xml_node& pnode, FLOAT* alpha) const;
    void from_XML_node(pugi::xml_node& pnode);
};


#endif // CLASSIFIER_H
