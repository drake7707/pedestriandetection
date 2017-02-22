#include "FeatureSet.h"



FeatureSet::FeatureSet()
{
}


FeatureSet::~FeatureSet()
{
}

void FeatureSet::addCreator(IFeatureCreator* creator) {
	this->creators.push_back(creator);
}

FeatureVector FeatureSet::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	if (creators.size() == 0)
		throw std::exception("No feature creates were present");

	FeatureVector v = creators[0]->getFeatures(rgb, depth);
	for (int i = 1; i < creators.size(); i++)
	{
		FeatureVector v2 = creators[i]->getFeatures(rgb, depth);
		for (auto& f : v2)
			v.push_back(f);
	}
	return v;
}

int FeatureSet::getNumberOfFeatures() const {
	int nrOfFeatures = 0;
	for (auto & c : creators)
		nrOfFeatures += c->getNumberOfFeatures();
	return nrOfFeatures;
}

std::string FeatureSet::explainFeature(int featureIndex, double splitValue) const {
	int from = 0;
	for (auto & c : creators) {
		int nrOfFeatures = c->getNumberOfFeatures();
		int to = from + nrOfFeatures;

		if (featureIndex >= from && featureIndex < to) {
			return c->explainFeature(featureIndex - from, splitValue);
		}
		from += nrOfFeatures;
	}

	return "";
}