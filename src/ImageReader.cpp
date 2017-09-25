#include "ImageReader.h"

ImageReader::ImageReader(std::string file){
  this-> file = file;
  this-> c_img = nullptr;
  
  iff.open(file, std::fstream::in);
  
  if(!iff.good()){
    std::cerr << "Error opening file with no_faces img list" << std::endl;
    exit(1);
  }
}

bool ImageReader::set_next_image(bool verbose){
  

  if(c_img != nullptr)
    if(c_img->get_height()/2 >= SIZE && c_img->get_width()/2 >= SIZE){
      c_img->shadow_resize_half();
      this->i = this->j = 0;
      return true;
    }
      
  std::string line;
  if(!std::getline(iff, line)){
    iff.close();
    return false;
  }
  
  unsigned found = this->file.find_last_of("/\\");
  
  if(found == std::string::npos)  // Same directory
    found = 0;
  else
    ++found;
  
  line = file.substr(0, found) + line;
  
  // Open image, may crash if you don't have imagemagick
  cimg_library::CImg<FLOAT> image(line.c_str());
  
  if(verbose)
    std::cout << "Opened image " << line << std::endl << std::endl;
  
  int width = image.width();
  int height = image.height();
  int d = width * height;

  float* data = image.data();
  FLOAT* ndata = new FLOAT[d];

  // Convert to grayscale
  for(int i = 0; i < d; ++i){
    ndata[i] = 0.2126*data[i] + 0.7152*data[i+d] + 0.0722*data[i+2*d];
  }

  if(c_img != nullptr)
    delete[] c_img->data;  // Poor man's destructor
  c_img = new Image(height, width, ndata);
  
  this->i = this->j = 0;
  
  return true;
}

bool ImageReader::fill(FixedImage* imgs, int nImgs, Class* hts, CascadeClassifier& ccl){
  
  uint32_t n_renew_img = 0; 
  
  // Count number of images that need renewal
  for(int img = 0; img < nImgs; ++img)
    if(hts[img] == Class::NFACE)
      ++n_renew_img;
  
  Image** renew_img = new Image*[n_renew_img];
  
  int count = 0; 
  
  // Copy the image addresses to our local vector
  for(int img = 0; img < nImgs; ++img)
    if(hts[img] == Class::NFACE)
      renew_img[count++] = &(imgs[img]);
      
  uint32_t img = 0;

  if(c_img == nullptr)
    set_next_image(true);

  FixedImage aux;
  uint32_t n_img = 0; 
  do{
    for( ; i < c_img->get_height() - SIZE; i+=4, j=0)
      for( ; j < c_img->get_width() - SIZE; j+=4){
        renew_img[img]->copy_data(*c_img, i, j, i + SIZE, j + SIZE);
        renew_img[img]->normalize();
        renew_img[img]->to_integral(aux);
        if(ccl.classify(*(renew_img[img])) == Class::FACE){
          renew_img[img]->c = Class::NFACE;
          ++img;
          if(img >= n_renew_img){
            delete [] renew_img;
            return true;
          }
        }
      }
    
    std::cout << "Added " << img - n_img << " no-faces" << std::endl;
    std::cout << std::endl;

    n_img = img;
  }while(set_next_image(true));
  
  // We've run out of images
  delete [] renew_img;
  return false;
}

