#ifndef CASCADECLASSIFIER_H
#define CASCADECLASSIFIER_H

#include <vector>
#include <iostream>

#include "Global.h"
#include "ImageReader.h"
#include "StrongClassifier.h"

#define NCLSTS 2

class CascadeClassifier{
  std::vector<StrongClassifier> cls;
  Classifier ** wcls;
  uint32_t nwCls[NCLSTS];
  
  int nF, nNF;
  
public:

  CascadeClassifier(){};
  
  void gen_wclassifiers(short sx = SIZE, short sy = SIZE, bool v2 = false, bool h2 = true, 
                                                          bool v3 = true, bool h3 = true);
  void train(FixedImage* imgs, int nImgs, int nscls, int* nswcls, FLOAT* nfpr, std::string nflist); 
  
  void to_XML(std::string file);
  void from_XML(std::string file);
  
  Class classify(const Image &img);
  void test(FixedImage* imgs, uint32_t nImgs, Class* hts = nullptr);
};


#endif
