#ifndef IMAGEREADER_H
#define IMAGEREADER_H

#include "CImg.h"
#include "Image.h"
#include "Global.h"
#include "CascadeClassifier.h"

class CascadeClassifier;

class ImageReader{
  
  std::string file;
  std::fstream iff;
  
  Image* c_img;
  int i, j;
  
public:
  
  ImageReader(std::string file);

  bool set_next_image(bool verbose = false);
  bool fill(FixedImage* imgs, int nImgs, Class* hts, CascadeClassifier& ccl);
};

#endif
