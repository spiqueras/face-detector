#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdlib.h>

#include "Global.h"
#include "CascadeClassifier.h"

uint32_t nF = 0, nNF = 0;  // Number of faces / noFaces
uint32_t nImgs = 0;  // nf + nNF

struct Conf{
  std::string ffaces, fnfaces, fimgs, fcl;
  int nscls;
  int* nswcls;
  FLOAT* max_fpr;
};

// Load parameters from xml file
Conf getConf(char* conf_xml){  
  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_file(conf_xml);  

  if(!result){
    std::cerr << "Error opening xml file: " << conf_xml << std::endl;
    exit(-1);
  }

  Conf conf;

  pugi::xml_node node = doc.first_child();

  conf.ffaces = node.attribute("ffaces").value();
  conf.fnfaces = node.attribute("fnfaces").value();
  conf.fimgs = node.attribute("fimgs").value();
  conf.fcl = node.attribute("fcl").value();

  conf.nscls = node.attribute("ncls").as_int();
  conf.nswcls = new int[conf.nscls];
  conf.max_fpr = new float[conf.nscls];

  int i = 0;  
  for (pugi::xml_node scnode = node.first_child(); scnode; scnode = scnode.next_sibling(), ++i){
    conf.nswcls[i] = scnode.attribute("nwcls").as_int();
    conf.max_fpr[i] = scnode.attribute("max_fpr").as_float();
  }

  return conf;
}

FixedImage* imgsFromFile(std::string ff, std::string fnf){

  std::fstream iff (ff, std::fstream::in);    // Faces file
  std::fstream ifnf (fnf, std::fstream::in);  // No faces file
  std::string line;

  if(!iff.good()){
    std::cerr << "The text file " << ff << " couldn't be opened" << std::endl;
    return nullptr;
  }

  if(!ifnf.good()){
    std::cerr << "The text file " << fnf << " couldn't be opened" << std::endl;
    return nullptr;
  }

  nF = nNF = 0;

  // Compute number of faces
  while(std::getline(iff, line)){
    nF++;
  }

  // Compute number of no-faces
  while(std::getline(ifnf, line)){
    nNF++;
  }

  iff.close();
  ifnf.close();


  FixedImage* imgs = new FixedImage[nF + nNF];
  FixedImage aux;

  // Read the faces
  iff.open(ff, std::fstream::in);

  for(uint32_t i = 0; i < nF; ++i){
    for(int j = 0; j < SIZE*SIZE; ++j){
      iff >> imgs[i].data[j];
      imgs[i].c = Class::FACE;
    }
    
    imgs[i].to_integral(aux);
  }

  iff.close();

  // Read the no-faces
  ifnf.open(fnf, std::fstream::in);

  for(uint32_t i = nF; i < nF+nNF; ++i){
    for(int j = 0; j < SIZE*SIZE; ++j){
      ifnf >> imgs[i].data[j];
      imgs[i].c = Class::NFACE;
    }

    imgs[i].to_integral(aux);

  }

  ifnf.close();

  nImgs = nF + nNF;

  return imgs;
}


int main (int argc, char** argv) {
  if(argc != 2){
    std::cerr << "Usage: " << argv[0] << " conf_xml" << std::endl;
    exit(1);
  }

  // Get parameters
  Conf conf = getConf(argv[1]);
  FixedImage* imgs = imgsFromFile(conf.ffaces, conf.fnfaces);
  
  std::cout << "Number of images: " << nImgs << std::endl;

  if(imgs == nullptr)
    exit(1);
  
  CascadeClassifier ccl;
  ccl.gen_wclassifiers();  // default values are good
  ccl.train(imgs, nImgs, conf.nscls, conf.nswcls, conf.max_fpr, conf.fimgs);
  ccl.to_XML(const_cast<char *>(conf.fcl.c_str()));
  
  delete [] imgs;

  imgs = imgsFromFile(conf.ffaces, conf.fnfaces);

  std::cout << "Cascade classifier: testing "<< nImgs << " images" << " (" << nF << " faces, " << nNF << " no-faces)" << std::endl;
  ccl.test(imgs, nImgs);

  delete [] imgs;
  
  return 0;
}
