#include "CascadeClassifier.h"


void CascadeClassifier::gen_wclassifiers(short sx, short sy, bool v2, bool h2, bool v3, bool h3){

  int count = 0;


  nwCls[0] = 0;
  if(v2){
    if(sx%2 == 0)
      nwCls[0] += sx*sx*(sx+1)*(sx+2)/8;
    else
      nwCls[0] += sx*(sx-1)*(sx+1)*(sx+1)/8;
  }

  if(h2){
    if(sx%2 == 0)
      nwCls[0] += sx*sx*(sx+1)*(sx+2)/8;
    else
      nwCls[0] += sx*(sx-1)*(sx+1)*(sx+1)/8;
  }

  this->wcls = new Classifier*[NCLSTS];
  
  Z2Classifier* z2cls;
  this->wcls[0] = z2cls = new Z2Classifier[nwCls[0]];

  count = 0;
  Point pts[6];


  if(h2) // HORIZONTAL 2 ZONES
    for(short p1x = 0; p1x < sx; p1x++){
      pts[0].x = p1x;
      for(short p1y = 0; p1y < sy; p1y++){
        pts[0].y = p1y;
        for(short p2x = p1x + 1; p2x < sx; p2x+=2){
          pts[3].x = p2x;
          for(short p2y = p1y; p2y < sy; p2y++){
            pts[3].y = p2y;

            pts[1].x = (p2x + p1x) / 2;
            pts[1].y = p2y;

            pts[2].x = pts[1].x + 1;
            pts[2].y = p1y;

            z2cls[count].set_ps(pts);

            ++count;
          }

        }
      }
    }

  if(v2) // VERTICAL 2 ZONES 
    for(short p1x = 0; p1x < sx; p1x++){
      pts[0].x = p1x;
      for(short p1y = 0; p1y < sy; p1y++){
        pts[0].y = p1y;
        for(short p2x = p1x; p2x < sx; p2x++){
          pts[3].x = p2x;
          for(short p2y = p1y + 1; p2y < sy; p2y+=2){
            pts[3].y = p2y;
            
            pts[1].x = p2x;
            pts[1].y = (p2y + p1y) / 2;

            pts[2].x = p1x;
            pts[2].y = pts[1].y + 1;

            z2cls[count].set_ps(pts);

            ++count;
          }

        }
      }
    }



  nwCls[1] = 0;
  if(v3)
    nwCls[1] += 3*sx*(sx+1)*((sx/3)+1)*(sx/3)/4;

  if(h3)
    nwCls[1] += 3*sx*(sx+1)*((sx/3)+1)*(sx/3)/4;

  Z3Classifier* z3cls;
  this->wcls[1] = z3cls = new Z3Classifier[nwCls[1]];

  count = 0;

  if(h3) // HORIZONTAL 3 ZONES
    for(short p1x = 0; p1x < sx; p1x++){
      pts[0].x = p1x;
      for(short p1y = 0; p1y < sy; p1y++){
        pts[0].y = p1y;
        for(short p2x = p1x + 2; p2x < sx; p2x+=3){
          pts[5].x = p2x;
          for(short p2y = p1y; p2y < sy; p2y++){
            pts[5].y = pts[3].y = pts[1].y = p2y;
            pts[2].y = pts[4].y = p1y;

            pts[1].x = p1x + ((p2x - p1x) / 3);
            pts[2].x = pts[1].x + 1;

            pts[3].x = p1x + ( (2*(p2x - p1x)) / 3);
            pts[4].x = pts[3].x + 1;

            z3cls[count].set_ps(pts);

            ++count;
          }

        }
      }
    }

  if(v3) // VERTICAL 3 ZONES
    for(short p1x = 0; p1x < sx; p1x++){
      pts[0].x = p1x;
      for(short p1y = 0; p1y < sy; p1y++){
        pts[0].y = p1y;
        for(short p2x = p1x; p2x < sx; p2x++){
          pts[5].x = p2x;
          for(short p2y = p1y + 2; p2y < sy; p2y+=3){
            pts[5].y = p2y;

            pts[3].x = pts[1].x = p2x;
            pts[2].x = pts[4].x = p1x;

            pts[1].y = p1y + ((p2y - p1y) / 3);
            pts[2].y = pts[1].y + 1;

            pts[3].y = p1y + ( (2*(p2y - p1y)) / 3);
            pts[4].y = pts[3].y + 1;

            z3cls[count].set_ps(pts);

            ++count;
          }

        }
      }
    }

}

Class CascadeClassifier::classify(const Image &img){
  for(auto& scl: cls){
    if(scl.classify(img) == Class::NFACE)
      return Class::NFACE;
  }
  
  return Class::FACE;
}

void CascadeClassifier::test(FixedImage* imgs, uint32_t nImgs, Class* hts){
  Class ht;
  uint32_t fpos, fneg;
  fpos = fneg = 0;

  for(uint32_t nImg = 0; nImg < nImgs; ++nImg){
    
    ht = classify(imgs[nImg]);
    
    if(hts != nullptr)
      hts[nImg] = ht;
    
    if(ht == Class::FACE && ht != imgs[nImg].c)
      fpos++;
    else if(ht == Class::NFACE && ht != imgs[nImg].c)
      fneg++;
  }

  std::cout << "False positives: " << fpos << std::endl;
  std::cout << "False negatives: " << fneg << std::endl;
  std::cout << std::endl;
}

void CascadeClassifier::train(FixedImage* imgs, int nImgs, int nscls, int* nswcls, FLOAT* nfpr, std::string nflist){
  if(wcls == nullptr){
    std::cerr << "Error: gen_wclassifiers must be called before train" << std::cout;
    return;
  }
  
  ImageReader img_reader(nflist);
  Class* hts = new Class[nImgs];
  
  for(int i = 0; i < nscls; ++i){
    StrongClassifier scl;
    scl.adaBoost(imgs, nImgs, wcls, nwCls, NCLSTS, nswcls[i]);
    scl.opt_thr(imgs, nImgs, nfpr[i]);
    
    std::cout << "Strong classifier " << i << std::endl;
    scl.test(imgs, nImgs, hts);
    cls.push_back(scl);
    
    if(i < (nscls - 1) && !img_reader.fill(imgs, nImgs, hts, *this)){
      std::cerr << "Warning: not enough no-face images to complete cascade" << std::endl;
      return;
    }
  }
  
}


void CascadeClassifier::to_XML(std::string file){
  
  pugi::xml_document doc;
  pugi::xml_node bnode = doc.append_child("cascadeclassifier");
  bnode.append_attribute("ncls") = (uint32_t)this->cls.size();
  for(auto& scl: cls){
    scl.add_to_XML_node(bnode);
  }

  doc.save_file(file.c_str());
}


void CascadeClassifier::from_XML(std::string file){
  
  pugi::xml_document doc;
  
  pugi::xml_parse_result result = doc.load_file(file.c_str());
  
  if(!result){
    std::cerr << "Error opening xml file: " << file << std::endl;
    exit(-1);
  }
  
  pugi::xml_node bnode = doc.first_child();
  
  if(strcmp (bnode.name(),"cascadeclassifier") != 0){
    std::cerr << "Error reading xml file: " << file << std::endl;
    exit(-1);
  }
  
  for (pugi::xml_node scnode = bnode.first_child(); scnode; scnode = scnode.next_sibling()){
    StrongClassifier scl;
    if(!scl.from_XML_node(scnode)){
      std::cerr << "Error reading xml file: " << file << std::endl;
      exit(-1);
    }
    cls.push_back(scl);
  }
}
