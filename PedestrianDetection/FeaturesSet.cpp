#include "FeaturesSet.h"



FeaturesSet::FeaturesSet()
{
}


FeaturesSet::~FeaturesSet()
{
}

void FeaturesSet::addCreator(IFeatureCreator* creator) {
	this->creators.push_back(creator);
}

FeatureVector FeaturesSet::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	FeatureVector v = creators[0]->getFeatures(rgb, depth);
	for (int i = 1; i < creators.size(); i++)
	{
		FeatureVector v2 = creators[i]->getFeatures(rgb, depth);
		for (auto& f : v2)
			v.push_back(f);
	}
	return v;
}
