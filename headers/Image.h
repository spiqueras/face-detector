#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include <fstream>
#include <math.h>
#include "Global.h"

/* Point */
struct Point{
  short x;
  short y;
};


/* Class */
enum Class{
  NFACE = -1,
  FACE = 1
};


/* Image */
class Image{
protected:
  int sx, sy;

public:
  FLOAT* data;
  Class c;

  Image(){};
  Image(int sx, int sy, FLOAT* data);

  void set_p(int x, int y, FLOAT val);
  void set_p(Point &p, FLOAT val);

  FLOAT get_p(int x, int y) const;
  FLOAT get_p(const Point &p) const;
  FLOAT get_area(const Point &p1, const Point &p2) const;

  int get_height(){ return sx; };
  int get_width(){ return sy; };

  void normalize();
  void to_integral(Image &aux);

  void shadow_resize_half();  
  void copy_data(const Image& img, int x, int y, int x_max, int y_max);
};

class FixedImage: public Image{

public:
  FLOAT fdata[SIZE*SIZE];

  FixedImage(){sx = SIZE; sy = SIZE; data = fdata;};
  FixedImage(FLOAT* data);
};

#endif // IMAGE_H
