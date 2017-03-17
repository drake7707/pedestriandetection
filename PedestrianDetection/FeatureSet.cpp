#include "FeatureSet.h"
#include "VariableNumberFeatureCreator.h"



FeatureSet::FeatureSet()
	: creators(0)
{
}


FeatureSet::~FeatureSet()
{
}

void FeatureSet::addCreator(std::unique_ptr<IFeatureCreator> creator) {
	this->creators.push_back(std::move(creator));
}

FeatureVector FeatureSet::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	if (creators.size() == 0)
		throw std::exception("No feature creators were present");

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
	for(int i = 0; i < creators.size(); i++)
		nrOfFeatures += creators[i]->getNumberOfFeatures();
	return nrOfFeatures;
}

std::string FeatureSet::explainFeature(int featureIndex, double splitValue) const {
	int from = 0;
	for (auto&& c : creators) {
		int nrOfFeatures = c->getNumberOfFeatures();
		int to = from + nrOfFeatures;

		if (featureIndex >= from && featureIndex < to) {
			return c->explainFeature(featureIndex - from, splitValue);
		}
		from += nrOfFeatures;
	}

	return "";
}

void FeatureSet::prepare(TrainingDataSet& trainingDataSet, int trainingRound) {
	for (auto&& c : creators) {
		if (dynamic_cast<VariableNumberFeatureCreator*>(c.get()) != nullptr) {
			(dynamic_cast<VariableNumberFeatureCreator*>(c.get()))->prepare(trainingDataSet, trainingRound);
		}
	}
}