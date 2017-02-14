#ifndef WEAK_CLASSIFIER 
#define WEAK_CLASSIFIER 1
#include "adaboost.hpp"

namespace DM_AG{

  class WeakClassifierOne : public Classifier<int>
  {
  public:
    int analyze(const int& i) const  {
      
      if (i>50 && i<90) return 1;
      return -1;
    }
  };
  
  class WeakClassifierTwo : public Classifier<int>
  {
  public:
    int analyze(const int& i) const {
      
      if (i>80 && i<130) return 1;
      return -1;
    }
  };
  
  class WeakClassifierThree : public Classifier<int>
  {
  public:
    int analyze(const int& i) const  {
      
      if (i>90 && i<130) return 1;
      return -1;
    }
  };
  
  class WeakClassifierFour : public Classifier<int>
  {
  public:
    int analyze(const int& i) const {
      
      if (i>40 && i<150) return 1;
      return -1;
    }
  };
};
#endif
