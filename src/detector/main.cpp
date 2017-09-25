#include "CImg.h"
#include "Image.h"
#include "CascadeClassifier.h"

#include <iostream>
#include <vector>

using namespace cimg_library;


int main(int argc, char** argv) {
  if(argc != 3){
    std::cerr << "Usage: " << argv[0] << " img_file cl_file" << std::endl;
    exit(-1);
  }

  CImg<float> image(argv[1]);

  int width = image.width();
  int height = image.height();
  int d = width * height;

  float* data = image.data();
  float* ndata = new float[d];


  int size = image.size();
  if(size == 3*d)  // Convert to grayscale
    for(int i = 0; i < d; ++i)
      ndata[i] = 0.2126*data[i] + 0.7152*data[i+d] + 0.0722*data[i + 2*d];
  else
    for(int i = 0; i < d; ++i)
      ndata[i] = data[i];

  Image img(height, width, ndata);

  CascadeClassifier scl;
  scl.from_XML(argv[2]);
  
  std::vector<Point> faces;

  short i = 0;
  #if defined(_OPENMP)  // Avoid warnings
  #pragma omp parallel
  #endif
  {

  FixedImage fimg;
  FixedImage aux;  // to_integral

  #if defined(_OPENMP)
  #pragma omp for
  #endif

  for(i = 0; i < img.get_height() - SIZE; ++i)
    for(short j = 0; j < img.get_width() - SIZE; ++j){
      fimg.copy_data(img, i, j, i + SIZE, j + SIZE);
      fimg.normalize();
      fimg.to_integral(aux);
      if(scl.classify(fimg) == Class::FACE)
  #if defined(_OPENMP)
  #pragma omp critical
  #endif
        faces.push_back({i,j});
    }
  }

  std::cout << "Height: " << height << std::endl;
  std::cout << "Width: " << width << std::endl;

  if(size == d){
    float* cdata = new float[3*d];
    for(int i = 0; i < d; ++i)
      cdata[i] = cdata[i+d] = cdata[i + 2*d] = data[i];
    
    image.assign(cdata, width, height, 1, 3);
  }

  for(const auto& face: faces){
    for(int i = face.x; i < face.x + SIZE; ++i){
      image(face.y, i, 0) = 255.0;
      image(face.y, i, 1) = 0;
      image(face.y, i, 2) = 0;
      image(face.y + SIZE - 1, i, 0) = 255.0;
      image(face.y + SIZE - 1, i, 1) = 0;
      image(face.y + SIZE - 1, i, 2) = 0;
    }
    
    for(int j = face.y; j < face.y + SIZE; ++j){
      image(j, face.x, 0) = 255.0;
      image(j, face.x, 1) = 0;
      image(j, face.x, 2) = 0;
      image(j, face.x + SIZE - 1, 0) = 255.0;
      image(j, face.x + SIZE - 1, 1) = 0;
      image(j, face.x + SIZE - 1, 2) = 0;
    }
  }

  CImgDisplay main_disp(image,"Lena");
  while (!main_disp.is_closed()) {
    main_disp.wait();
    if(main_disp.is_resized())
      main_disp.resize(true);
  }
  return 0;
}

