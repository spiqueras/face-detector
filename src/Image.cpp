#include "Image.h"

//************ Image


// Constructor
Image::Image(int sx, int sy, FLOAT* data){
  this->sx = sx;
  this->sy = sy;
  this->data = new FLOAT[sx*sy];

  for(int i = 0; i < sx*sy; i++)
    this->data[i] = data[i];
  
}

// Get and set points
void Image::set_p(int x, int y, FLOAT val){
  if (! (x < 0 || y < 0))
    data[x * this->sy + y] = val;
}

void Image::set_p(Point &p, FLOAT val){
  set_p(p.x, p.y, val);
}

FLOAT Image::get_p(int x, int y) const{
  if (x < 0 || y < 0)
    return 0;
  return data[x * this->sy + y];
}

FLOAT Image::get_p(const Point &p) const{
  return get_p(p.x, p.y);
}

// Computes the integral image
void Image::to_integral(Image &aux){

  for(int i = 0; i < sx; ++i)
    for(int j = 0; j < sy; ++j){
      aux.set_p(i, j, aux.get_p(i, j-1) + this->get_p(i, j) );
      this->set_p(i, j, this->get_p(i-1, j) + aux.get_p(i, j) );
    }
}

void Image::normalize(){
  
  FLOAT mean = 0, var = 0, std;
  for(int i = 0; i < sx*sy; ++i)
    mean += data[i];
  mean /= sx*sy;
  
  
  for(int i = 0; i < sx*sy; ++i)
    var += (data[i] - mean) * (data[i] - mean);
  var /= sx*sy;
  std = sqrt(var);
  
  for(int i = 0; i < sx*sy; ++i){
    data[i] = (data[i] - mean) / std;
  }
}

// Get the area contained between 2 points
// Requires the image to be integral
FLOAT Image::get_area(const Point &p1, const Point &p2) const{
  short i = p1.x-1;
  short j = p1.y-1;
  return get_p(p2) + get_p(i, j)
          - (get_p(i, p2.y) + get_p(p2.x, j));
}


void Image::shadow_resize_half(){
  int sx = this->sx / 2;
  int sy = this->sy / 2;
  for(int i = 0; i < sx; ++i)
    for(int j = 0; j < sy; ++j)
      data[i*sy + j] = (data[i*2 * this->sy + j*2]
                      + data[i*2 * this->sy + (j*2 + 1)] 
                      + data[(i*2+1) * this->sy + j*2]
                      + data[(i*2+1) * this->sy + (j*2 + 1)]) / 4.0;
      // data(i,j) = data(i*2,j*2) + data(i*2,j*2+1) + data(i*2 + 1,j*2) + data(i*2 + 1, j*2 + 1);

  this->sx = sx;
  this->sy = sy;
}


void Image::copy_data(const Image& img, int x, int y, int x_max, int y_max){
  for(int i = x; i < x_max; ++i)
    for(int j = y; j < y_max; ++j)
      set_p(i-x, j-y, img.get_p(i,j));

}

//************ FixedImage

FixedImage::FixedImage(FLOAT* data){

  this->sx = SIZE;
  this->sy = SIZE;

  for(int i = 0; i < SIZE*SIZE; i++)
    this->fdata[i] = data[i];

  this->data = fdata;

}
