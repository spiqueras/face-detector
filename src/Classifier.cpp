#include "Classifier.h"
#define POW232 4294967296


/* Classifier */

FLOAT Classifier::get_e_opt_thr(FixedImage* imgs, uint32_t nImgs, FLOAT* D){
  
  FLOAT sp, sn, tp, tn;
  sp = sn = tp = tn = 0;

  Othr_aux* es = new Othr_aux[nImgs];

  for(uint32_t nImg = 0; nImg < nImgs; ++nImg){
    es[nImg].i = nImg;
    es[nImg].val = get_val(imgs[nImg]);

    if(imgs[nImg].c == Class::FACE)
      tp += D[nImg];
    else
      tn += D[nImg];
  }

  std::sort(es, es + nImgs, [](const Othr_aux& o1, 
                               const Othr_aux& o2){
                                  return (o1.val < o2.val);
                            }
           );

  FLOAT e = 0, e1 = 0, e2 = 0;
  FLOAT min_e = 1;
  FLOAT thr = this->thr;
  int p;

  for(uint32_t nImg = 0; nImg < nImgs; ++nImg){
    if(imgs[es[nImg].i].c == Class::FACE)
      sp += D[es[nImg].i];
    else
      sn += D[es[nImg].i];

    e1 = sp + (tn - sn);
    e2 = sn + (tp - sp);

    if(e1 < e2){
      e = e1;
      p = 1;
    }else{
      e = e2;
      p = -1;
    }

    if(e < min_e){
      min_e = e;
      thr = es[nImg].val;
      this->p = p;
    }
  }

  this->thr = thr;

  delete [] es;

  return min_e;
}


Class Classifier::classify(const Image &img) const{

  return (get_val(img)*p >= thr*p)? 
          Class::FACE : 
          Class::NFACE;

}


/* Z2Classifier */


// Copy constructor
Z2Classifier::Z2Classifier(const Z2Classifier& other){

  this->ps = other.ps;
  this->p = other.p;
  this->thr = other.thr;
}


void Z2Classifier::set_ps(Point* ps){

  this->ps.z11 = ps[0];
  this->ps.z12 = ps[1];
  this->ps.z21 = ps[2];
  this->ps.z22 = ps[3];
}


FLOAT Z2Classifier::get_val(const Image &img) const{

  return (img.get_area(ps.z11,ps.z12) - img.get_area(ps.z21,ps.z22));

}


void Z2Classifier::print() const{

  if(p == 1)
      std::cout << "WB classifier" << std::endl;
  else
      std::cout << "BW classifier" << std::endl;

  std::cout << "First zone goes from " << ps.z11.x << "," << ps.z11.y << " to " 
                                        << ps.z12.x << "," << ps.z12.y << std::endl;
  std::cout << "Second zone goes from " << ps.z21.x << "," << ps.z21.y << " to " 
                                        << ps.z22.x << "," << ps.z22.y << std::endl;

}


void Z2Classifier::add_to_XML_node(pugi::xml_node& pnode, FLOAT* alpha) const{
	pugi::xml_node bnode = pnode.append_child("wclassifier");
	bnode.append_attribute("type") = "Z2";
	bnode.append_attribute("polarity") = this->p;
	bnode.append_attribute("threshold") = this->thr;
	bnode.append_attribute("alpha_pos") = alpha[0];
	bnode.append_attribute("alpha_neg") = alpha[1];
	
	pugi::xml_node auxnode = bnode.append_child("z11");
	auxnode.append_attribute("x") = ps.z11.x;
	auxnode.append_attribute("y") = ps.z11.y;
	
	auxnode = bnode.append_child("z12");
	auxnode.append_attribute("x") = ps.z12.x;
	auxnode.append_attribute("y") = ps.z12.y;
	
	auxnode = bnode.append_child("z21");
	auxnode.append_attribute("x") = ps.z21.x;
	auxnode.append_attribute("y") = ps.z21.y;
	
	auxnode = bnode.append_child("z22");
	auxnode.append_attribute("x") = ps.z22.x;
	auxnode.append_attribute("y") = ps.z22.y;
	
}


void Z2Classifier::from_XML_node(pugi::xml_node& pnode){
	this->p = pnode.attribute("polarity").as_int();
	this->thr = pnode.attribute("threshold").as_double();
	
  for (pugi::xml_node node = pnode.first_child(); node; node = node.next_sibling()){
    if (strcmp(node.name(), "z11") == 0){
      ps.z11.x = node.attribute("x").as_int();
      ps.z11.y = node.attribute("y").as_int();
    }else if (strcmp(node.name(), "z12") == 0){
      ps.z12.x = node.attribute("x").as_int();
      ps.z12.y = node.attribute("y").as_int();
    }else	if (strcmp(node.name(), "z21") == 0){
      ps.z21.x = node.attribute("x").as_int();
      ps.z21.y = node.attribute("y").as_int();
    }else if (strcmp(node.name(), "z22") == 0){
      ps.z22.x = node.attribute("x").as_int();
      ps.z22.y = node.attribute("y").as_int();
    }
	}
}


/* Z3Classifier */


// Copy constructor
Z3Classifier::Z3Classifier(const Z3Classifier& other){

  this->ps = other.ps;
  this->p = other.p;
  this->thr = other.thr;
}


void Z3Classifier::set_ps(Point* ps){

  this->ps.z11 = ps[0];
  this->ps.z12 = ps[1];

  this->ps.z21 = ps[2];
  this->ps.z22 = ps[3];

  this->ps.z31 = ps[4];
  this->ps.z32 = ps[5];
}


FLOAT Z3Classifier::get_val(const Image &img) const{

  return ((img.get_area(ps.z11,ps.z12) + img.get_area(ps.z31,ps.z32))
           - img.get_area(ps.z21,ps.z22));

}


void Z3Classifier::print() const{

  if(p == 1)
      std::cout << "WBW classifier" << std::endl;
  else
      std::cout << "BWB classifier" << std::endl;

  std::cout << "First zone goes from " << ps.z11.x << "," << ps.z11.y << " to " 
                                        << ps.z12.x << "," << ps.z12.y << std::endl;
  std::cout << "Second zone goes from " << ps.z21.x << "," << ps.z21.y << " to " 
                                        << ps.z22.x << "," << ps.z22.y << std::endl;
  std::cout << "Third zone goes from " << ps.z31.x << "," << ps.z31.y << " to " 
                                        << ps.z32.x << "," << ps.z32.y << std::endl;

}


void Z3Classifier::add_to_XML_node(pugi::xml_node& pnode, FLOAT* alpha) const{
	pugi::xml_node bnode = pnode.append_child("wclassifier");
	bnode.append_attribute("type") = "Z3";
	bnode.append_attribute("polarity") = this->p;
	bnode.append_attribute("threshold") = this->thr;
	bnode.append_attribute("alpha_pos") = alpha[0];
	bnode.append_attribute("alpha_neg") = alpha[1];
	
	pugi::xml_node auxnode = bnode.append_child("z11");
	auxnode.append_attribute("x") = ps.z11.x;
	auxnode.append_attribute("y") = ps.z11.y;
	
	auxnode = bnode.append_child("z12");
	auxnode.append_attribute("x") = ps.z12.x;
	auxnode.append_attribute("y") = ps.z12.y;
	
	auxnode = bnode.append_child("z21");
	auxnode.append_attribute("x") = ps.z21.x;
	auxnode.append_attribute("y") = ps.z21.y;
	
	auxnode = bnode.append_child("z22");
	auxnode.append_attribute("x") = ps.z22.x;
	auxnode.append_attribute("y") = ps.z22.y;
	
	auxnode = bnode.append_child("z31");
	auxnode.append_attribute("x") = ps.z31.x;
	auxnode.append_attribute("y") = ps.z31.y;
	
	auxnode = bnode.append_child("z32");
	auxnode.append_attribute("x") = ps.z32.x;
	auxnode.append_attribute("y") = ps.z32.y;
}


void Z3Classifier::from_XML_node(pugi::xml_node& pnode){
	this->p = pnode.attribute("polarity").as_int();
	this->thr = pnode.attribute("threshold").as_double();
	
  for (pugi::xml_node node = pnode.first_child(); node; node = node.next_sibling()){
    if (strcmp(node.name(), "z11") == 0){
      ps.z11.x = node.attribute("x").as_int();
      ps.z11.y = node.attribute("y").as_int();
    }else if (strcmp(node.name(), "z12") == 0){
      ps.z12.x = node.attribute("x").as_int();
      ps.z12.y = node.attribute("y").as_int();
    }else	if (strcmp(node.name(), "z21") == 0){
      ps.z21.x = node.attribute("x").as_int();
      ps.z21.y = node.attribute("y").as_int();
    }else if (strcmp(node.name(), "z22") == 0){
      ps.z22.x = node.attribute("x").as_int();
      ps.z22.y = node.attribute("y").as_int();
    }else if (strcmp(node.name(), "z31") == 0){
      ps.z31.x = node.attribute("x").as_int();
      ps.z31.y = node.attribute("y").as_int();
    }else if (strcmp(node.name(), "z32") == 0){
      ps.z32.x = node.attribute("x").as_int();
      ps.z32.y = node.attribute("y").as_int();
    }
	}
}
