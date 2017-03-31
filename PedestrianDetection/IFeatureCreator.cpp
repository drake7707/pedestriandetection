#include "IFeatureCreator.h"



IFeatureCreator::IFeatureCreator(std::string& name) : name(name) {
}


IFeatureCreator::~IFeatureCreator() {
}


std::string IFeatureCreator::getName() const {
	return name;
}

