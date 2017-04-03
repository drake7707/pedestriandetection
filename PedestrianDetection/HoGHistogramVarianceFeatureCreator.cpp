#include "HoGHistogramVarianceFeatureCreator.h"
#include "HistogramOfOrientedGradients.h"



HOGHistogramVarianceFeatureCreator::HOGHistogramVarianceFeatureCreator(std::string& name, IFeatureCreator::Target target, int patchSize, int binSize, int refWidth, int refHeight)
	: IFeatureCreator(name), patchSize(patchSize), binSize(binSize), refWidth(refWidth), refHeight(refHeight), target(target) {
}


HOGHistogramVarianceFeatureCreator::~HOGHistogramVarianceFeatureCreator()
{
}


int HOGHistogramVarianceFeatureCreator::getNumberOfFeatures() const {
	// each rectangle of 2x2 histograms, each containing binSize elements is now replaced by a single S2 value
	return hog::getNumberOfFeatures(refWidth, refHeight, patchSize, binSize, true) / (binSize * 4);
}

FeatureVector HOGHistogramVarianceFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const {
	hog::HistogramResult result;
	if (target == IFeatureCreator::Target::Depth)
		result = hog::getHistogramsOfOrientedGradient(depth, patchSize, binSize, false, true);
	else if(target == IFeatureCreator::Target::Thermal)
		result = hog::getHistogramsOfOrientedGradient(thermal, patchSize, binSize, false, true);
	else
		result = hog::getHistogramsOfOrientedGradient(rgb, patchSize, binSize, false, true);

	return result.getHistogramVarianceFeatures();
}


cv::Mat HOGHistogramVarianceFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	int nrOfCellsWidth = refWidth / patchSize;
	int nrOfCellsHeight = refHeight / patchSize;

	int nrOfFeatures = getNumberOfFeatures();
	bool l2normalize = true;

	cv::Mat explanation(cv::Size(refWidth, refHeight), CV_32FC1, cv::Scalar(0));

	std::function<void(int, int, int)> func = [&](int featureIndex, int patchX, int patchY) -> void {
		int x = patchX * patchSize;
		int y = patchY * patchSize;
		double weight = occurrencePerFeature[offset + featureIndex];
		cv::rectangle(explanation, cv::Rect(x, y, patchSize, patchSize), cv::Scalar(weight), -1);
	};


	int idx = 0;
	if (l2normalize) {
		for (int y = 0; y < nrOfCellsHeight - 1; y++) {
			for (int x = 0; x < nrOfCellsWidth - 1; x++) {


				func(idx, x, y);
				idx++;

				func(idx, x + 1, y);
				idx++;

				func(idx, x, y + 1);
				idx++;
				func(idx, x + 1, y + 1);
				idx++;
			}
		}
	}
	else {
		for (int y = 0; y < nrOfCellsHeight; y++) {
			for (int x = 0; x < nrOfCellsWidth; x++) {
				func(idx, x, y);
				idx++;
			}
		}
	}
	return explanation;
}

std::vector<bool> HOGHistogramVarianceFeatureCreator::getRequirements() const {
	return{ target == IFeatureCreator::Target::RGB,
		target == IFeatureCreator::Target::Depth,
		target == IFeatureCreator::Target::Thermal
	};
}