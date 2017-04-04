#include "IEvaluator.h"
#include <mutex>
#include <thread>
#include "ProgressWindow.h"
#include "Helper.h"
#include <stack>
#include <map>


IEvaluator::IEvaluator(std::string& name)
	: name(name)
{
}



IEvaluator::~IEvaluator()
{
}


float IEvaluator::getValueShift(int i, int nrOfEvaluations, float evaluationRange) const {
	double valueShift = 1.0 * i / nrOfEvaluations * evaluationRange - evaluationRange / 2;
	return valueShift;
}

std::vector<ClassifierEvaluation> IEvaluator::evaluateDataSet(const TrainingDataSet& trainingDataSet, const FeatureSet& set, const EvaluationSettings& settings, std::function<bool(int imageNumber)> canSelectFunc) {
	std::vector<ClassifierEvaluation> evals(settings.nrOfEvaluations, ClassifierEvaluation());

	double sumTimes = 0;
	int nrRegions = 0;
	double featureBuildTime = 0;

	trainingDataSet.iterateDataSet(canSelectFunc,
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string(name), 1.0 * imageNumber / trainingDataSet.getNumberOfImages(), std::string("Evaluating training set regions (") + std::to_string(imageNumber) + ")");

		// don't use prepared data and roi for training set
		std::vector<IPreparedData*> preparedData;
		cv::Rect roi;

		FeatureVector v;
		featureBuildTime += measure<std::chrono::milliseconds>::execution([&]() -> void {
			v = set.getFeatures(rgb, depth, thermal, roi, preparedData);
		});


		double resultSum;
		sumTimes += measure<std::chrono::milliseconds>::execution([&]() -> void {
			resultSum = evaluateFeatures(v);
		});
		nrRegions++;
		for (int i = 0; i < settings.nrOfEvaluations; i++)
		{
			// get datapoints, ranging from -evaluationRange/2 to evaluationRange/2
			double valueShift = getValueShift(i, settings.nrOfEvaluations, settings.evaluationRange);
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
		}
	}, settings.addFlippedInTrainingSet, settings.refWidth, settings.refHeight);

	for (int i = 0; i < settings.nrOfEvaluations; i++)
		evals[i].evaluationSpeedPerRegionMS = (featureBuildTime + sumTimes) / nrRegions;

	ProgressWindow::getInstance()->finish(std::string(name));

	return evals;
}


EvaluationSlidingWindowResult IEvaluator::evaluateWithSlidingWindow(const EvaluationSettings& settings,
	const DataSet* dataSet, const FeatureSet& set,
	std::function<bool(int number)> canSelectFunc) {

	EvaluationSlidingWindowResult swresult;
	swresult.evaluations = std::vector<ClassifierEvaluation>(settings.nrOfEvaluations, ClassifierEvaluation(dataSet->getNrOfImages()));

	double sumTimesRegions = 0;
	int nrRegions = 0;
	double featureBuildTime = 0;

	auto comp = [](std::pair<float, SlidingWindowRegion> a, std::pair<float, SlidingWindowRegion> b) { return a.first > b.first; };
	//std::priority_queue<std::pair<float, SlidingWindowRegion>, std::vector<std::pair<float, SlidingWindowRegion>>, decltype(comp)> worstFalseNegatives(comp);

	std::mutex mutex;
	std::function<void(std::function<void()>)> lock = [&](std::function<void()> func) -> void {
		mutex.lock();
		func();
		mutex.unlock();
	};;

	int maxNrOfFPPerImage = settings.maxNrOfFPPerImage;

	typedef std::vector<std::set<SlidingWindowRegion>> FPPerValueShift;

	std::vector<FPPerValueShift> worstFalsePositivesArrayPerImage(dataSet->getNrOfImages(),
		FPPerValueShift(settings.nrOfEvaluations, std::set<SlidingWindowRegion>()));


	std::map<int, std::vector<std::vector<IPreparedData*>>> preparedDataPerImage;

	int nrOfImagesEvaluated = 0;
	dataSet->iterateDataSetWithSlidingWindow(settings.windowSizes, settings.baseWindowStride, settings.refWidth, settings.refHeight,
		canSelectFunc,
		[&](int imgNr, std::vector<cv::Mat>& rgbScales, std::vector<cv::Mat>& depthScales, std::vector<cv::Mat>& thermalScales) -> void {
		// image has started

		std::vector<std::vector<IPreparedData*>> preparedData = set.buildPreparedDataForFeatures(rgbScales, depthScales, thermalScales);
		lock([&]() -> void {
			preparedDataPerImage[imgNr] = std::move(preparedData);
		});
	},
		[&](int idx, int resultClass, int imageNumber, int scale, cv::Rect& scaledRegion, cv::Rect& unscaledROI, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal, bool overlapsWithTruePositive) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string(name), 1.0 * nrOfImagesEvaluated / dataSet->getNrOfImages(), std::string("Evaluating with sliding window (") + std::to_string(nrOfImagesEvaluated) + "/" + std::to_string(dataSet->getNrOfImages()) + ")");

		FeatureVector v;
		long buildTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
			v = set.getFeatures(rgb, depth, thermal, unscaledROI, preparedDataPerImage[imageNumber][scale]);
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
			for (int i = 0; i < settings.nrOfEvaluations; i++)
			{
				double valueShift = getValueShift(i, settings.nrOfEvaluations, settings.evaluationRange);
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
						swresult.evaluations[i].falsePositivesPerImage[imageNumber]++;
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

							auto& worstFalsePositivesOfImageAndEvaluationShift = worstFalsePositivesArrayPerImage[imageNumber][i];
							if (worstFalsePositivesOfImageAndEvaluationShift.size() == 0 || abs(evaluationResult) > abs(worstFalsePositivesOfImageAndEvaluationShift.begin()->score)) {
								// if the new false positive is better than the smallest element

								// check if the element has any overlap with the elements currently in the queue
								bool overlapsWithElement = false;
								auto it = worstFalsePositivesOfImageAndEvaluationShift.rbegin();
								while (it != worstFalsePositivesOfImageAndEvaluationShift.rend()) {
									if (getIntersectionOverUnion(it->bbox, scaledRegion) > 0.5) {
										overlapsWithElement = true;

										if (abs(evaluationResult) > abs(it->score)) {
											// swap in this element
											worstFalsePositivesOfImageAndEvaluationShift.erase(std::next(it).base()); //(http://stackoverflow.com/questions/1830158/how-to-call-erase-with-a-reverse-iterator)
											worstFalsePositivesOfImageAndEvaluationShift.insert(SlidingWindowRegion(imageNumber, scaledRegion, abs(evaluationResult)));
										}
										else {
											// smaller result, the worst element is already in the queue
										}
										// the set is iterated in reverse, so biggest first. If it was overlapping the element could either be replaced or already have 
										// a smaller result. As all other elements will have a smaller result we can stop now.
										break;
									}
									it++;
								}

								// prevent false positive with a big result and overlap from completely filling the priority queue so disallow elements that overlap enough (IoU > 0.5)
								if (!overlapsWithElement) {
									// add it 
									worstFalsePositivesOfImageAndEvaluationShift.emplace(SlidingWindowRegion(imageNumber, scaledRegion, abs(evaluationResult)));

									// smallest results will be removed to ensure not exceeding the maximum capacity
									if (worstFalsePositivesOfImageAndEvaluationShift.size() > maxNrOfFPPerImage) // keep the top worst performing regions
										worstFalsePositivesOfImageAndEvaluationShift.erase(worstFalsePositivesOfImageAndEvaluationShift.begin());
								}
							}
						}
						else {
							//worstFalseNegatives.push(std::pair<float, SlidingWindowRegion>(abs(evaluationResult), SlidingWindowRegion(imageNumber, region)));
							//// smallest numbers will be popped
							//if (worstFalseNegatives.size() > maxNrOfFalsePosOrNeg) // keep the top worst performing regions
							//	worstFalseNegatives.pop();
						}
					}
				}
			}
		});
	}, [&](int imgNr, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositiveRegions) -> void {
		// full image is processed
		lock([&]() -> void {
			// clean up of prepared data
			for (auto& v : preparedDataPerImage[imgNr]) {
				for (int i = 0; i < v.size(); i++) {
					if (v[i] != nullptr)
						delete v[i];
				}
			}
			preparedDataPerImage.erase(imgNr);

			for (int i = 0; i < settings.nrOfEvaluations; i++)
				swresult.evaluations[i].nrOfImagesEvaluated++;
			nrOfImagesEvaluated++;
		});
	}, settings.slidingWindowParallelization);

	// iteration with multiple threads is done, update the evaluation timings and the worst false positive/negatives

	for (int i = 0; i < settings.nrOfEvaluations; i++)
		swresult.evaluations[i].evaluationSpeedPerRegionMS = 1.0 * (featureBuildTime + sumTimesRegions) / nrRegions;

	swresult.worstFalsePositives.reserve(settings.maxNrOfFalsePosOrNeg);
	//swresult.worstFalseNegatives.reserve(worstFalseNegatives.size());

	// determine value shift where TPR = tprToObtainWorstFalsePositives
	double valueShiftRequired = std::numeric_limits<double>().min();
	double minTPRAboveTPRToObtainWorstFalsePositives = std::numeric_limits<double>().max();;
	int evaluationIndexForValueShift = -1;

	std::cout << std::fixed;
	for (int i = 0; i < swresult.evaluations.size(); i++)
	{
		if (swresult.evaluations[i].getTPR() > settings.requiredTPRRate) { // only consider evaluations above the min threshold TPR
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
	// still not enough, just take the largest value
	if (evaluationIndexForValueShift == -1) {
		evaluationIndexForValueShift = swresult.evaluations.size() - 1;
		minTPRAboveTPRToObtainWorstFalsePositives = swresult.evaluations[evaluationIndexForValueShift].getTPR();
		valueShiftRequired = swresult.evaluations[evaluationIndexForValueShift].valueShift;
	}

	swresult.evaluationIndexWhenTPRThresholdIsReached = evaluationIndexForValueShift;

	std::cout << "Chosen " << valueShiftRequired << " as decision boundary shift to attain TPR of " << swresult.evaluations[evaluationIndexForValueShift].getTPR() << std::endl;


	// for each image in the dataset a worst false positive queue per value shift is filled in
	// the value shift and corresponding index is know, so just iterate all the images
	// and put them into 1 final priority queue, hard capped with the max nr of FP to add to the training set
	// This way each image has a chanche to present a few false positives and only the worst of their worst ones will be kept

	std::set<SlidingWindowRegion> worstFalsePositives = std::set<SlidingWindowRegion>();
	for (auto& worstFalsePositiveOfImage : worstFalsePositivesArrayPerImage) {

		for (auto& wfp : worstFalsePositiveOfImage[evaluationIndexForValueShift]) {
			worstFalsePositives.emplace(wfp);
		}
	}

	int nrAdded = 0;
	auto it = worstFalsePositives.rbegin();
	while (it != worstFalsePositives.rend() && nrAdded < settings.maxNrOfFalsePosOrNeg) {
		auto& wnd = *it;
		//std::cout << "Worst FP region score= " << wnd.score << " image= " << wnd.imageNumber << " bbox=" << wnd.bbox.x << "," << wnd.bbox.y << " " << wnd.bbox.width << "x" << wnd.bbox.height << std::endl;
		swresult.worstFalsePositives.push_back(wnd);
		nrAdded++;
		it++;
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


FinalEvaluationSlidingWindowResult IEvaluator::evaluateWithSlidingWindowAndNMS(const EvaluationSettings& settings,
	const DataSet* dataSet, const FeatureSet& set, std::function<bool(int number)> canSelectFunc) {

	FinalEvaluationSlidingWindowResult swresult;

	for (auto& category : dataSet->getCategories())
		swresult.evaluations[category] = std::vector<ClassifierEvaluation>(settings.nrOfEvaluations, ClassifierEvaluation(dataSet->getNrOfImages()));
	swresult.combinedEvaluations = std::vector<ClassifierEvaluation>(settings.nrOfEvaluations, ClassifierEvaluation(dataSet->getNrOfImages()));

	double sumTimesRegions = 0;
	int nrRegions = 0;
	double featureBuildTime = 0;

	std::mutex mutex;
	std::function<void(std::function<void()>)> lock = [&](std::function<void()> func) -> void {
		mutex.lock();
		func();
		mutex.unlock();
	};;

	std::map<int, std::vector<SlidingWindowRegion>> evaluatedWindowsPerImage;
	std::map<int, std::vector<std::vector<IPreparedData*>>> preparedDataPerImage;
	int nrOfImagesEvaluated = 0;
	dataSet->iterateDataSetWithSlidingWindow(settings.windowSizes, settings.baseWindowStride, settings.refWidth, settings.refHeight,
		canSelectFunc,
		[&](int imgNr, std::vector<cv::Mat>& rgbScales, std::vector<cv::Mat>& depthScales, std::vector<cv::Mat>& thermalScales) -> void {
		// image has started

		std::vector<std::vector<IPreparedData*>> preparedData = set.buildPreparedDataForFeatures(rgbScales, depthScales, thermalScales);
		lock([&]() -> void {
			swresult.missedPositivesPerImage[imgNr] = std::vector<std::vector<cv::Rect>>(settings.nrOfEvaluations);
			evaluatedWindowsPerImage[imgNr] = std::vector<SlidingWindowRegion>();
			preparedDataPerImage[imgNr] = std::move(preparedData);
		});
	},
		[&](int idx, int resultClass, int imageNumber, int scale, cv::Rect& scaledRegion, cv::Rect& unscaledROI, cv::Mat&rgb, cv::Mat&depth, cv::Mat& thermal, bool overlapsWithTruePositive) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string(name), 1.0 * nrOfImagesEvaluated / dataSet->getNrOfImages(), std::string("Evaluating with sliding window and NMS (") + std::to_string(nrOfImagesEvaluated) + ")");


		FeatureVector v;
		long buildTime = measure<std::chrono::milliseconds>::execution([&]() -> void {
			v = set.getFeatures(rgb, depth, thermal, unscaledROI, preparedDataPerImage[imageNumber][scale]);
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
		evaluatedWindowsPerImage[imageNumber].push_back(SlidingWindowRegion(imageNumber, scaledRegion, baseEvaluationSum));

	}, [&](int imgNr, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositiveRegions) -> void {
		// full image is processed

		auto& windows = evaluatedWindowsPerImage[imgNr];

		for (int i = 0; i < settings.nrOfEvaluations; i++) {
			double valueShift = getValueShift(i, settings.nrOfEvaluations, settings.evaluationRange);

			// get the predicted positives
			std::vector<SlidingWindowRegion> predictedPositives;
			for (auto& w : windows) {
				int resultClass = (w.score + valueShift) > 0 ? 1 : -1;
				if (resultClass == 1)
					predictedPositives.push_back(SlidingWindowRegion(w.imageNumber, w.bbox, abs(w.score + valueShift)));
			}
			predictedPositives = applyNonMaximumSuppression(predictedPositives);

			lock([&]() -> void {
				for (auto& category : dataSet->getCategories())
					swresult.evaluations[category][i].valueShift = valueShift;

				swresult.combinedEvaluations[i].valueShift = valueShift;

				int tp = 0;
				for (auto& predpos : predictedPositives) {

					int tpIndex = getOverlapIndex(predpos.bbox, truePositiveRegions);
					if (tpIndex != -1) {
						//  predicted positive and true positive
						if (swresult.evaluations.find(truePositiveCategories[tpIndex]) != swresult.evaluations.end())
							swresult.evaluations[truePositiveCategories[tpIndex]][i].nrOfTruePositives++;

						swresult.combinedEvaluations[i].nrOfTruePositives++;
					}
					else {
						for (auto& category : dataSet->getCategories()) {
							swresult.evaluations[category][i].nrOfFalsePositives++;
							swresult.evaluations[category][i].falsePositivesPerImage[imgNr]++;
						}

						swresult.combinedEvaluations[i].nrOfFalsePositives++;
						swresult.combinedEvaluations[i].falsePositivesPerImage[imgNr]++;
					}
				}

				std::vector<cv::Rect2d> predictedPosRegions;
				for (auto& r : predictedPositives)
					predictedPosRegions.push_back(r.bbox);

				for (int tp = 0; tp < truePositiveRegions.size(); tp++)
				{
					if (!overlaps(truePositiveRegions[tp], predictedPosRegions)) {
						// missed a true positive
						if (swresult.evaluations.find(truePositiveCategories[tp]) != swresult.evaluations.end())
							swresult.evaluations[truePositiveCategories[tp]][i].nrOfFalseNegatives++;
						swresult.combinedEvaluations[i].nrOfFalseNegatives++;

						swresult.missedPositivesPerImage[imgNr][i].push_back(truePositiveRegions[tp]);
					}
				}
			});
		}
		lock([&]() -> void {
			// done with image
			evaluatedWindowsPerImage.erase(imgNr);

			// clean up of prepared data
			for (auto& v : preparedDataPerImage[imgNr]) {
				for (int i = 0; i < v.size(); i++) {
					if (v[i] != nullptr)
						delete v[i];
				}
			}
			preparedDataPerImage.erase(imgNr);

			for (int i = 0; i < settings.nrOfEvaluations; i++) {
				for (auto& category : dataSet->getCategories())
					swresult.evaluations[category][i].nrOfImagesEvaluated++;

				swresult.combinedEvaluations[i].nrOfImagesEvaluated++;
			}
			nrOfImagesEvaluated++;
		});
	}, settings.slidingWindowParallelization);

	// iteration with multiple threads is done, update the evaluation timings and the worst false positive/negatives

	for (int i = 0; i < settings.nrOfEvaluations; i++) {

		for (auto& category : dataSet->getCategories())
			swresult.evaluations[category][i].evaluationSpeedPerRegionMS = 1.0 * (featureBuildTime + sumTimesRegions) / nrRegions;

		swresult.combinedEvaluations[i].evaluationSpeedPerRegionMS = 1.0 * (featureBuildTime + sumTimesRegions) / nrRegions;
	}

	ProgressWindow::getInstance()->finish(std::string(name));

	return swresult;
}