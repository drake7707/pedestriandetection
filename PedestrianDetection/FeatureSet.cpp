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

int FeatureSet::size() const {
	return this->creators.size();
}


std::vector<std::vector<IPreparedData*>> FeatureSet::buildPreparedDataForFeatures(std::vector<cv::Mat>& rgbScales, std::vector<cv::Mat>& depthScales, std::vector<cv::Mat>& thermalScales) const {
	if (creators.size() == 0)
		throw std::exception("No feature creators were present");

	int nrOfScales = rgbScales.size();

	std::vector<std::vector<IPreparedData*>> preparedDataPerScalePerCreator(nrOfScales, std::vector<IPreparedData*>());
	
	for (int i = 0; i < creators.size(); i++) {
		auto preparedDataPerScale = creators[i]->buildPreparedDataForFeatures(rgbScales, depthScales, thermalScales);
		for (int s = 0; s < preparedDataPerScale.size(); s++)
			preparedDataPerScalePerCreator[s].push_back(preparedDataPerScale[s]);
	}
	
	return preparedDataPerScalePerCreator;
}

FeatureVector FeatureSet::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal, cv::Rect& roi, const std::vector<IPreparedData*>& preparedDataOfScale) const {
	if (creators.size() == 0)
		throw std::exception("No feature creators were present");

	FeatureVector v = creators[0]->getFeatures(rgb, depth, thermal, roi, preparedDataOfScale.size() > 0 ? preparedDataOfScale.at(0) : nullptr);
	for (int i = 1; i < creators.size(); i++)
	{
		FeatureVector v2 = creators[i]->getFeatures(rgb, depth, thermal, roi, preparedDataOfScale.size() > 0 ? preparedDataOfScale.at(i) : nullptr);
		for (auto& f : v2)
			v.push_back(f);
	}
	return v;
}

int FeatureSet::getNumberOfFeatures() const {
	int nrOfFeatures = 0;
	for (int i = 0; i < creators.size(); i++)
		nrOfFeatures += creators[i]->getNumberOfFeatures();
	return nrOfFeatures;
}

std::vector<cv::Mat> FeatureSet::explainFeatures(std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	int from = 0;

	std::vector<cv::Mat> explainImages(size());

	int i = 0;
	for (auto&& c : creators) {
		int nrOfFeatures = c->getNumberOfFeatures();
		int to = from + nrOfFeatures;

		cv::Mat img = c->explainFeatures(from, weightPerFeature, occurrencePerFeature, refWidth, refHeight);
		explainImages[i] = img;

		from += nrOfFeatures;
		i++;
	}
	return explainImages;
}

void FeatureSet::prepare(TrainingDataSet& trainingDataSet, const EvaluationSettings& settings) {
	for (auto&& c : creators) {
		if (dynamic_cast<VariableNumberFeatureCreator*>(c.get()) != nullptr) {
			(dynamic_cast<VariableNumberFeatureCreator*>(c.get()))->prepare(trainingDataSet, settings);
		}
	}
}

std::vector<bool> FeatureSet::getRequirements() const {
	std::vector<bool> v(3, false);
	for (auto&& c : creators) {
		auto requirementsOfCreator = c->getRequirements();
		for (int i = 0; i < requirementsOfCreator.size(); i++)
		{
			bool req = requirementsOfCreator[i];
			v[i] = v[i] || req;
		}
	}
	return v;
}