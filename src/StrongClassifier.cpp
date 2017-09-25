#include "StrongClassifier.h"

void StrongClassifier::opt_thr(FixedImage* imgs, uint32_t nImgs, FLOAT max_fnr, bool verbose){
  
  FLOAT* thrs = new FLOAT[nImgs];
  FLOAT* vals = new FLOAT[nImgs];
  uint32_t npos, nneg;
  npos = nneg = 0;
  
  for(uint32_t nImg = 0; nImg < nImgs; ++nImg){
    thrs[nImg] = vals[nImg] = get_val(imgs[nImg]);
    (imgs[nImg].c == Class::FACE) ? ++npos : ++nneg; 
  }

  std::sort(thrs, thrs + nImgs);
  auto it = std::unique(thrs, thrs + nImgs);
  int nThrs =  std::distance(thrs, it);;

  Class ht;

  for(int i = 0; i < nThrs; ++i){
    uint32_t fpos, fneg;
    fpos = fneg = 0;
    thrs[i] -= 0.0001; // Avoid rounding problems (really)

    for(uint32_t nImg = 0; nImg < nImgs; ++nImg){
      ht = (vals[nImg] >= thrs[i]) ? Class::FACE : Class::NFACE;
      
      if(ht == Class::FACE && ht != imgs[nImg].c)
        fpos++;
      else if(ht == Class::NFACE && ht != imgs[nImg].c)
        fneg++;
    }
    
    if(verbose){
      std::cout << "Testing: threshold = " << thrs[i] << std::endl;
      std::cout << "False positives ratio: " << 100.0*(float)fpos/nneg << "%" << std::endl;
      std::cout << "False negatives ratio: " << 100.0*(float)fneg/npos << "%" << std::endl;
      std::cout << std::endl;
    }
      

    if((float)fneg/npos < max_fnr){
      if(verbose){
        std::cout << std::endl;
      }
      
      this->thr = thrs[i]; 
    }else{
      std::cout << "Threshold set to " << this-> thr << std::endl;
      break;
    }
  }

  
  delete [] thrs;
  delete [] vals;
}


FLOAT StrongClassifier::get_val(const Image &img){
  FLOAT acc = 0;
  Class ht;

  for(int i = 0; i < nCls; ++i){
    ht = ccls[i]->classify(img);
    acc += ht * alpha[i][ht == Class::FACE ? 0 : 1];
  }

  return acc;
}


Class StrongClassifier::classify(const Image &img){    
  return (get_val(img) >= thr) ? Class::FACE : Class::NFACE;
}


void StrongClassifier::test(FixedImage* imgs, uint32_t nImgs, Class* hts){
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


void StrongClassifier::adaBoost(FixedImage* imgs, uint32_t nImgs, Classifier** cls, uint32_t* nCls, uint32_t NCLSTS, int T){

  // Create and initialize D
  FLOAT* D = new FLOAT[nImgs];
  std::fill(D, D + nImgs, 1.0/(FLOAT)nImgs);

  // Compute total number of classifiers
  uint32_t ntCls = 0;
  for(uint32_t i = 0; i < NCLSTS; i++)
    ntCls += nCls[i];

  // Create vector with errors
  FLOAT* es = new FLOAT[ntCls];

  FLOAT zt = 0;
  Class ht;

  this->ccls = new Classifier*[T];
  this->nCls = T;
  this->alpha = new FLOAT*[T];
  for(int i = 0; i < T; i++)
    alpha[i] = new FLOAT[2];

  uint32_t fpos, fneg;

  for (int aux = 0; aux < T; ++aux){
    
    std::cout << std::endl << "Iteration " << aux << std::endl;

    for(uint32_t i = 0; i < NCLSTS; ++i){
      uint32_t d = 0;
      for(uint32_t j = 0; j < i; ++j)
        d += nCls[j];

      uint32_t nCl;

      #if defined(_OPENMP)  // Avoid warnings
      #pragma omp parallel for
      #endif
      for(nCl = 0; nCl < nCls[i]; ++nCl){

        if (i == 0)
          es[d + nCl] = (static_cast<Z2Classifier*>(cls[i]))[nCl].get_e_opt_thr(imgs, nImgs, D);
        else if(i==1)
          es[d + nCl] = (static_cast<Z3Classifier*>(cls[i]))[nCl].get_e_opt_thr(imgs, nImgs, D);

      }
    }

    auto pet = std::min_element(es, es + ntCls);  // Pointer to the min_element
    auto t = std::distance(es, pet);

    uint32_t i = 0;
    while(t > nCls[i]){
      t-=nCls[i];
      i++;
    }

    FLOAT et = *pet;

    // Copy classifiers (in order to keep the threshold/polarity)
    if (i == 0)
      ccls[aux] = new Z2Classifier(static_cast<Z2Classifier*>(cls[i])[t]);
    else if(i==1)
      ccls[aux] = new Z3Classifier(static_cast<Z3Classifier*>(cls[i])[t]);


    // Compute alpha for the classifier t
    alpha[aux][0] = alpha[aux][1] = log((1-et)/et) / 2;


    // Compute the new image weights
    fpos = fneg = 0;
    zt = 0;

    for(uint32_t nImg = 0; nImg < nImgs; ++nImg){
      ht = ccls[aux]->classify(imgs[nImg]);
      zt += D[nImg] = D[nImg] * exp(- alpha[aux][0] * imgs[nImg].c * ht);
      
      if(ht == Class::FACE && ht != imgs[nImg].c)
        fpos++;
      else if(ht == Class::NFACE && ht != imgs[nImg].c)
        fneg++;
    }

    // Normalize the weights
    for(uint32_t nImg = 0; nImg < nImgs; ++nImg)
      D[nImg] = D[nImg] / zt;


    ccls[aux]->print();
    std::cout << "The value of et is " << et << std::endl;
    std::cout << "The value of alphat is " << alpha[aux][0] << std::endl;
    std::cout << "False positives " << fpos << std::endl;
    std::cout << "False negatives " << fneg << std::endl;

    std::cout << std::endl;
  
    if(et < 0.0001){  // Avoid float rounding errors
      this->nCls = aux + 1;
      break;
    }

  }

  std::cout << std::endl;

}


void StrongClassifier::rareBoost(FixedImage* imgs, uint32_t nImgs, Classifier** cls, uint32_t* nCls, uint32_t NCLSTS, int T){
  
  // Create and initialize D
  FLOAT* D = new FLOAT[nImgs];
  std::fill(D, D + nImgs, 1.0/(FLOAT)nImgs);

  uint32_t ntCls = 0;
  for(uint32_t i = 0; i < NCLSTS; i++)
    ntCls += nCls[i];

  // Create vector with errors
  FLOAT* es = new FLOAT[ntCls];
  this->alpha = new FLOAT*[T];
  for(int i = 0; i < T; i++)
    alpha[i] = new FLOAT[2];

  this->ccls = new Classifier*[T];
  this->nCls = T;

  FLOAT zt = 0;
  Class ht;

  uint32_t fpos, fneg;

  for (int aux = 0; aux < T; ++aux){
    
    std::cout << std::endl << "Iteration " << aux << std::endl;
    
    for(uint32_t i = 0; i < NCLSTS; ++i){
      uint32_t d = 0;
      for(uint32_t j = 0; j < i; ++j)
        d += nCls[j];

      uint32_t nCl;

      #if defined(_OPENMP)  // Avoid warnings
      #pragma omp parallel for
      #endif
      for(nCl = 0; nCl < nCls[i]; ++nCl){

        if (i == 0)
          es[d + nCl] = (static_cast<Z2Classifier*>(cls[i]))[nCl].get_e_opt_thr(imgs, nImgs, D);
        else if(i==1)
          es[d + nCl] = (static_cast<Z3Classifier*>(cls[i]))[nCl].get_e_opt_thr(imgs, nImgs, D);

      }
    }

    auto pet = std::min_element(es, es + ntCls);  // Pointer to the min_element
    auto t = std::distance(es, pet);

    uint32_t i = 0;
    while(t > nCls[i]){
      t-=nCls[i];
      i++;
    }

    FLOAT et = *pet;
    
    // Copy classifiers (in order to keep the threshold/polarity)
    if (i == 0)
      ccls[aux] = new Z2Classifier(static_cast<Z2Classifier*>(cls[i])[t]);
    else if(i==1)
      ccls[aux] = new Z3Classifier(static_cast<Z3Classifier*>(cls[i])[t]);

    FLOAT FNw, FPw, TNw, TPw;
    FNw = FPw = TNw = TPw = 0.0000001;

    // Compute alpha for the classifier t
    
    for(uint32_t nImg = 0; nImg < nImgs; ++nImg){
      
      ht = ccls[aux]->classify(imgs[nImg]);

      if(ht == imgs[nImg].c){
        if(ht == Class::NFACE)
          TNw += D[nImg];  // True negatives
        else 
          TPw += D[nImg];  // True positives
      }else{
        if(ht == Class::FACE)
          FPw += D[nImg];  // False positives
        else 
          FNw += D[nImg];  // False negatives
      }
    }
    
    alpha[aux][0] = log( TPw / FPw ) / 2;  // Weight for ht == FACE
    alpha[aux][1] = log( TNw / FNw ) / 2;  // Weight for ht == NFACE

    fpos = fneg = 0;

    // Compute the new image weights
    zt = 0;
    for(uint32_t nImg = 0; nImg < nImgs; ++nImg){
      ht = ccls[aux]->classify(imgs[nImg]);

      // Check which alpha[aux][i] we need to use
      int i = 0;
      if(ht == Class::NFACE)
        i = 1;
      else
        i = 0;

      FLOAT alpha_ = alpha[aux][i];

      zt += D[nImg] = D[nImg] * exp(- alpha_ * imgs[nImg].c * ht);
      
      if(ht == Class::FACE && ht != imgs[nImg].c)
        ++fpos;
      else if(ht == Class::NFACE && ht != imgs[nImg].c)
        ++fneg;
    }

    // Normalize the weights
    for(uint32_t nImg = 0; nImg < nImgs; ++nImg)
      D[nImg] = D[nImg] / zt;


    ccls[aux]->print();
    std::cout << "The value of et is " << et << std::endl;
    std::cout << "The value of alphat[0] is " << alpha[aux][0] << std::endl;
    std::cout << "The value of alphat[1] is " << alpha[aux][1] << std::endl;
    std::cout << "False positives " << fpos << std::endl;
    std::cout << "False negatives " << fneg << std::endl;

  }

  std::cout << std::endl;

}


void StrongClassifier::to_XML(std::string file) const{
  
  pugi::xml_document doc;
  add_to_XML_node(doc);

  doc.save_file(file.c_str());
}

void StrongClassifier::from_XML(std::string file){
  
  pugi::xml_document doc;
  
  pugi::xml_parse_result result = doc.load_file(file.c_str());
  
  if(!result){
    std::cerr << "Error opening xml file: " << file << std::endl;
    exit(-1);
  }
  
  pugi::xml_node pnode = doc.first_child();
    
  if(!from_XML_node(pnode)){
    std::cerr << "Error reading xml file: " << file << std::endl;
    exit(-1);
  }
    
}

bool StrongClassifier::from_XML_node(pugi::xml_node& pnode){
  
	if(strcmp (pnode.name(),"strongclassifier") == 0){
    for (pugi::xml_attribute attr = pnode.first_attribute(); attr; attr = attr.next_attribute())
      if(strcmp(attr.name(),"threshold") == 0)
        this->thr = attr.as_double();
      else if(strcmp(attr.name(), "nwcls") ==0)
        this->nCls = attr.as_int();
      else
        std::cerr << "Warning: unrecognised attribute " << attr.name() << std::endl;
  }else
    return false;
      
  this->ccls = new Classifier*[nCls];
  this->alpha = new FLOAT*[nCls];
  for(int i = 0; i < nCls; ++i)
    alpha[i] = new FLOAT[2];

  std::string type;
  int i = 0;
  
  for (pugi::xml_node wcnode = pnode.first_child(); wcnode; wcnode = wcnode.next_sibling()){
    this->alpha[i][0] = wcnode.attribute("alpha_pos").as_double();
    this->alpha[i][1] = wcnode.attribute("alpha_neg").as_double();
    type = wcnode.attribute("type").value();
    
    if(type == "Z2"){
      ccls[i] = new Z2Classifier();
      ccls[i]->from_XML_node(wcnode);
    }else if(type == "Z3"){
      ccls[i] = new Z3Classifier();
      ccls[i]->from_XML_node(wcnode);
    }else{
      std::cerr << "Warning: unrecognised type " << type << std::endl;
      nCls--;
    }
    ++i;
  }
	
  return true;
}


void StrongClassifier::add_to_XML_node(pugi::xml_node& pnode) const{
  pugi::xml_node bnode = pnode.append_child("strongclassifier");
  bnode.append_attribute("threshold") = this->thr;
  bnode.append_attribute("nwcls") = this->nCls;
  
  for(int i = 0; i < nCls; ++i){
    ccls[i]->add_to_XML_node(bnode, alpha[i]);
  }
}
