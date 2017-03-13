#include "IEvaluator.h"
#include <mutex>
#include <thread>
#include "ProgressWindow.h"
#include "Helper.h"


IEvaluator::IEvaluator(std::string& name)
	: name(name)
{
}



IEvaluator::~IEvaluator()
{
}



std::vector<ClassifierEvaluation> IEvaluator::evaluateDataSet(const TrainingDataSet& trainingDataSet, const FeatureSet& set,int nrOfEvaluations, bool includeRawResponses, std::function<bool(int imageNumber)> canSelectFunc) const {
	std::vector<ClassifierEvaluation> evals(nrOfEvaluations, ClassifierEvaluation());

	double sumTimes = 0;
	int nrRegions = 0;
	double featureBuildTime = 0;

	trainingDataSet.iterateDataSet(canSelectFunc,
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string(name), 1.0 * imageNumber / trainingDataSet.getNumberOfImages(), std::string("Evaluating training set regions (") + std::to_string(imageNumber) + ")");

		FeatureVector v;
		featureBuildTime += measure<std::chrono::milliseconds>::execution([&]() -> void {
			v = set.getFeatures(rgb, depth);
		});


		double resultSum;
		sumTimes += measure<std::chrono::milliseconds>::execution([&]() -> void {
			resultSum = evaluateFeatures(v);
		});
		nrRegions++;
		for (int i = 0; i < nrOfEvaluations; i++)
		{
			// get datapoints, ranging from -15 to 15
			double valueShift = 1.0 * i / nrOfEvaluations * evaluationRange - evaluationRange / 2;
			evals[i].valueShift = valueShift;

			int evaluationClass = resultSum + valueShift > 0 ? 1 : -1;

			bool correct;
			if (resultClass == evaluationClass) {
				if (resultClass == -1)
					evals[i].nrOfTrueNegatives++;
				else
					evals[i].nrOfTruePositives++;

				correct = true;
			}
			else {
				if (resultClass == -1 && evaluationClass == 1)
					evals[i].nrOfFalsePositives++;
				else
					evals[i].nrOfFalseNegatives++;

				correct = false;
			}

			if (includeRawResponses)
				evals[i].rawValues.push_back(RawEvaluationEntry(imageNumber, region, resultSum + valueShift, correct));


		}
	});

	for (int i = 0; i < nrOfEvaluations; i++)
		evals[i].evaluationSpeedPerRegionMS = (featureBuildTime + sumTimes) / nrRegions;

	ProgressWindow::getInstance()->finish(std::string(name));

	return evals;
}


EvaluationSlidingWindowResult IEvaluator::evaluateWithSlidingWindow(const TrainingDataSet& trainingDataSet, const FeatureSet& set,int nrOfEvaluations, int trainingRound, float tprToObtainWorstFalsePositives, int maxNrOfFalsePosOrNeg) const {

	EvaluationSlidingWindowResult swresult;
	swresult.evaluations = std::vector<ClassifierEvaluation>(nrOfEvaluations, ClassifierEvaluation());

	double sumTimesRegions = 0;
	int nrRegions = 0;
	double featureBuildTime = 0;

	auto comp = [](std::pair<float, SlidingWindowRegion> a, std::pair<float, SlidingWindowRegion> b) { return a.first > b.first; };
	std::vector<std::priority_queue<std::pair<float, SlidingWindowRegion>, std::vector<std::pair<float, SlidingWindowRegion>>, decltype(comp)>> worstFalsePositivesArray(nrOfEvaluations,
		std::priority_queue<std::pair<float, SlidingWindowRegion>, std::vector<std::pair<float, SlidingWindowRegion>>, decltype(comp)>(comp));
	//std::priority_queue<std::pair<float, SlidingWindowRegion>, std::vector<std::pair<float, SlidingWindowRegion>>, decltype(comp)> worstFalseNegatives(comp);

	std::mutex mutex;
	std::function<void(std::function<void()>)> lock = [&](std::function<void()> func) -> void {
		mutex.lock();
		func();
		mutex.unlock();
	};;


	trainingDataSet.iterateDataSetWithSlidingWindow([&](int idx) -> bool { return (idx + trainingRound) % slidingWindowEveryXImage == 0; },
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb, bool overlapsWithTruePositive) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string(name), 1.0 * imageNumber / trainingDataSet.getNumberOfImages(), std::string("Evaluating with sliding window (") + std::to_string(imageNumber) + ")");

		FeatureVector v;
		long buildTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
			v = set.getFeatures(rgb, depth);
		});

		lock([&]() -> void { // lock everything that is outside the function body as the iteration is done over multiple threads
			featureBuildTime += buildTime;
			nrRegions++;
		});

		// just evaluate it once and shift it
		double baseEvaluationSum;
		long time = measure<std::chrono::milliseconds>::execution([&]() -> void {
			baseEvaluationSum = evaluateFeatures(v);
		});

		// now update everything that is shared across threads from the built vectors
		lock([&]() -> void {
			for (int i = 0; i < nrOfEvaluations; i++)
			{
				double valueShift = 1.0 * i / nrOfEvaluations * evaluationRange - evaluationRange / 2;
				swresult.evaluations[i].valueShift = valueShift;

				double evaluationResult = baseEvaluationSum + valueShift;
				double evaluationClass = evaluationResult > 0 ? 1 : -1;
				bool correct;
				if (resultClass == evaluationClass) {
					if (resultClass == -1)
						swresult.evaluations[i].nrOfTrueNegatives++;
					else {
						swresult.evaluations[i].nrOfTruePositives++;
					}

					correct = true;
				}
				else {
					if (resultClass == -1 && evaluationClass == 1) {
						swresult.evaluations[i].nrOfFalsePositives++;
					}
					else
						swresult.evaluations[i].nrOfFalseNegatives++;

					correct = false;
				}

				// don't consider adding hard negatives or positives if it overlaps with a true positive
				// because it will add samples to the negative set that are really really strong and should probably be considered positive
				if (!overlapsWithTruePositive) {
					// only add worst false positives if valueShift is the value that was tested on and it was incorrect
					if (!correct) {
						if (evaluationClass == 1) {

							worstFalsePositivesArray[i].push(std::pair<float, SlidingWindowRegion>(abs(evaluationResult), SlidingWindowRegion(imageNumber, region)));
							// smallest numbers will be popped
							if (worstFalsePositivesArray[i].size() > maxNrOfFalsePosOrNeg) // keep the top 1000 worst performing regions
								worstFalsePositivesArray[i].pop();
						}
						else {
							//worstFalseNegatives.push(std::pair<float, SlidingWindowRegion>(abs(evaluationResult), SlidingWindowRegion(imageNumber, region)));
							//// smallest numbers will be popped
							//if (worstFalseNegatives.size() > maxNrOfFalsePosOrNeg) // keep the top 1000 worst performing regions
							//	worstFalseNegatives.pop();
						}
					}
				}

			}
		});
	});

	// iteration with multiple threads is done, update the evaluation timings and the worst false positive/negatives


	for (int i = 0; i < nrOfEvaluations; i++)
		swresult.evaluations[i].evaluationSpeedPerRegionMS = 1.0 * (featureBuildTime + sumTimesRegions) / nrRegions;

	swresult.worstFalsePositives.reserve(maxNrOfFalsePosOrNeg);
	//swresult.worstFalseNegatives.reserve(worstFalseNegatives.size());

	// determine value shift where TPR = tprToObtainWorstFalsePositives
	double valueShiftRequired = std::numeric_limits<double>().min();
	double minTPRAboveTPRToObtainWorstFalsePositives = std::numeric_limits<double>().max();;
	int evaluationIndexForValueShift = -1;

	std::cout << std::fixed;
	for (int i = 0; i < swresult.evaluations.size(); i++)
	{
		if (swresult.evaluations[i].getTPR() > tprToObtainWorstFalsePositives) { // only consider evaluations above the min threshold TPR
			if (swresult.evaluations[i].getTPR() < minTPRAboveTPRToObtainWorstFalsePositives) { // then take the smallest TPR above the threshold to reduce the FPR
				minTPRAboveTPRToObtainWorstFalsePositives = swresult.evaluations[i].getTPR();
				evaluationIndexForValueShift = i;
				valueShiftRequired = swresult.evaluations[i].valueShift;
			}
		}
	}
	if (evaluationIndexForValueShift == -1) {
		// there is no evaluation with TPR >= treshold
		// take the largest TPR value then
		double maxTPR = std::numeric_limits<double>().min();
		for (int i = 0; i < swresult.evaluations.size(); i++)
		{
			if (swresult.evaluations[i].getTPR() > maxTPR) { // then take the smallest TPR above the threshold to reduce the FPR
				minTPRAboveTPRToObtainWorstFalsePositives = swresult.evaluations[i].getTPR();
				evaluationIndexForValueShift = i;
				valueShiftRequired = swresult.evaluations[i].valueShift;
				maxTPR = swresult.evaluations[i].getTPR();
			}
		}
	}
	swresult.evaluationIndexWhenTPRThresholdIsReached = evaluationIndexForValueShift;

	std::cout << "Chosen " << valueShiftRequired << " as decision boundary shift to attain TPR of " << swresult.evaluations[evaluationIndexForValueShift].getTPR() << std::endl;

	while (worstFalsePositivesArray[evaluationIndexForValueShift].size() > 0) {
		auto pair = worstFalsePositivesArray[evaluationIndexForValueShift].top();
		worstFalsePositivesArray[evaluationIndexForValueShift].pop();
		std::cout << "Worst FP region score= " << pair.first << " image= " << pair.second.imageNumber << " bbox=" << pair.second.bbox.x << "," << pair.second.bbox.y << " " << pair.second.bbox.width << "x" << pair.second.bbox.height << std::endl;
		swresult.worstFalsePositives.push_back(pair.second);
	}

	//while (worstFalseNegatives.size() > 0) {
	//	auto pair = worstFalseNegatives.top();
	//	worstFalseNegatives.pop();
	//	//	std::cout << "Worst FN region score= " << pair.first << " image= " << pair.second.imageNumber << " bbox=" << pair.second.bbox.x << "," << pair.second.bbox.y << " " << pair.second.bbox.width << "x" << pair.second.bbox.height << std::endl;
	//	swresult.worstFalseNegatives.push_back(pair.second);
	//}

	ProgressWindow::getInstance()->finish(std::string(name));

	return swresult;
}
//
//double IEvaluator::evaluateWindow(cv::Mat& rgb, cv::Mat& depth) const {
//
//	FeatureVector v = set.getFeatures(rgb, depth);
//	return evaluateFeatures(v);
//}