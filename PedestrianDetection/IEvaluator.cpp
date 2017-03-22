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



std::vector<ClassifierEvaluation> IEvaluator::evaluateDataSet(const TrainingDataSet& trainingDataSet, const FeatureSet& set, int nrOfEvaluations, bool includeRawResponses, std::function<bool(int imageNumber)> canSelectFunc) const {
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
			// get datapoints, ranging from -evaluationRange/2 to evaluationRange/2
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


EvaluationSlidingWindowResult IEvaluator::evaluateWithSlidingWindow(std::vector<cv::Size>& windowSizes,
	const DataSet* dataSet, const FeatureSet& set, int nrOfEvaluations, int trainingRound,
	float tprToObtainWorstFalsePositives, int maxNrOfFalsePosOrNeg,
	std::function<bool(int number)> canSelectFunc) const {

	EvaluationSlidingWindowResult swresult;
	swresult.evaluations = std::vector<ClassifierEvaluation>(nrOfEvaluations, ClassifierEvaluation(dataSet->getNrOfImages()));

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



	// want to obtain 4000 false positives, over 7500 images, which means about 0.5 false positive per image
	// allow 20 times that size, or 10 worst false positives per image
	int maxNrOfFPPerImage = dataSet->getNrOfImages() / maxNrOfFalsePosOrNeg * 20;



	typedef std::vector<std::set<SlidingWindowRegion>> FPPerValueShift;

	std::vector<FPPerValueShift> worstFalsePositivesArrayPerImage(dataSet->getNrOfImages(),
		FPPerValueShift(nrOfEvaluations, std::set<SlidingWindowRegion>()));


	int nrOfImagesEvaluated = 0;
	dataSet->iterateDataSetWithSlidingWindow(windowSizes, baseWindowStride, refWidth, refHeight,
		canSelectFunc,
		[&](int imgNr) -> void {
		// image has started
	},
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb, bool overlapsWithTruePositive) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string(name), 1.0 * nrOfImagesEvaluated / dataSet->getNrOfImages(), std::string("Evaluating with sliding window (") + std::to_string(nrOfImagesEvaluated) + "/" + std::to_string(dataSet->getNrOfImages()) + ")");


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
									if (getIntersectionOverUnion(it->bbox, region) > 0.5) {
										overlapsWithElement = true;

										if (abs(evaluationResult) > abs(it->score)) {
											// swap in this element
											worstFalsePositivesOfImageAndEvaluationShift.erase(std::next(it).base()); //(http://stackoverflow.com/questions/1830158/how-to-call-erase-with-a-reverse-iterator)
											worstFalsePositivesOfImageAndEvaluationShift.insert(SlidingWindowRegion(imageNumber, region, abs(evaluationResult)));
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
									worstFalsePositivesOfImageAndEvaluationShift.emplace(SlidingWindowRegion(imageNumber, region, abs(evaluationResult)));

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
			for (int i = 0; i < nrOfEvaluations; i++)
				swresult.evaluations[i].nrOfImagesEvaluated++;
			nrOfImagesEvaluated++;
		});
	}, parallelization);

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
	while (it != worstFalsePositives.rend() && nrAdded < maxNrOfFalsePosOrNeg) {
		auto& wnd = *it;
		std::cout << "Worst FP region score= " << wnd.score << " image= " << wnd.imageNumber << " bbox=" << wnd.bbox.x << "," << wnd.bbox.y << " " << wnd.bbox.width << "x" << wnd.bbox.height << std::endl;
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


FinalEvaluationSlidingWindowResult IEvaluator::evaluateWithSlidingWindowAndNMS(std::vector<cv::Size>& windowSizes,
	const DataSet* dataSet, const FeatureSet& set, int nrOfEvaluations, std::function<bool(int number)> canSelectFunc,
	int refWidth, int refHeight, int paralellization) const {

	FinalEvaluationSlidingWindowResult swresult;
	// TODO don't hardcode categories
	swresult.evaluations["easy"] = std::vector<ClassifierEvaluation>(nrOfEvaluations, ClassifierEvaluation(dataSet->getNrOfImages()));
	swresult.evaluations["moderate"] = std::vector<ClassifierEvaluation>(nrOfEvaluations, ClassifierEvaluation(dataSet->getNrOfImages()));
	swresult.evaluations["hard"] = std::vector<ClassifierEvaluation>(nrOfEvaluations, ClassifierEvaluation(dataSet->getNrOfImages()));
	swresult.combinedEvaluations = std::vector<ClassifierEvaluation>(nrOfEvaluations, ClassifierEvaluation(dataSet->getNrOfImages()));
	double sumTimesRegions = 0;
	int nrRegions = 0;
	double featureBuildTime = 0;

	std::mutex mutex;
	std::function<void(std::function<void()>)> lock = [&](std::function<void()> func) -> void {
		mutex.lock();
		func();
		mutex.unlock();
	};;

	std::map<int, std::vector<SlidingWindowRegion>> map;

	int nrOfImagesEvaluated = 0;
	dataSet->iterateDataSetWithSlidingWindow(windowSizes, baseWindowStride, refWidth, refHeight,
		canSelectFunc,
		[&](int imgNr) -> void {
		// image has started
		lock([&]() -> void {
			map[imgNr] = std::vector<SlidingWindowRegion>();
		});
	},
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth, cv::Mat& fullrgb, bool overlapsWithTruePositive) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(std::string(name), 1.0 * nrOfImagesEvaluated / dataSet->getNrOfImages(), std::string("Evaluating with sliding window and NMS (") + std::to_string(nrOfImagesEvaluated) + ")");


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
		map[imageNumber].push_back(SlidingWindowRegion(imageNumber, region, baseEvaluationSum));

	}, [&](int imgNr, std::vector<std::string>& truePositiveCategories, std::vector<cv::Rect2d>& truePositiveRegions) -> void {
		// full image is processed

		auto& windows = map[imgNr];

		for (int i = 0; i < nrOfEvaluations; i++) {
			double valueShift = 1.0 * i / nrOfEvaluations * evaluationRange - evaluationRange / 2;

			// get the predicted positives
			std::vector<SlidingWindowRegion> predictedPositives;
			for (auto& w : windows) {
				int resultClass = (w.score + valueShift) > 0 ? 1 : -1;
				if (resultClass == 1)
					predictedPositives.push_back(SlidingWindowRegion(w.imageNumber, w.bbox, abs(w.score + valueShift)));
			}
			predictedPositives = applyNonMaximumSuppression(predictedPositives);

			lock([&]() -> void {
				swresult.evaluations["easy"][i].valueShift = valueShift;
				swresult.evaluations["moderate"][i].valueShift = valueShift;
				swresult.evaluations["hard"][i].valueShift = valueShift;
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
						swresult.evaluations["easy"][i].nrOfFalsePositives++;
						swresult.evaluations["easy"][i].falsePositivesPerImage[imgNr]++;
						swresult.evaluations["moderate"][i].nrOfFalsePositives++;
						swresult.evaluations["moderate"][i].falsePositivesPerImage[imgNr]++;
						swresult.evaluations["hard"][i].nrOfFalsePositives++;
						swresult.evaluations["hard"][i].falsePositivesPerImage[imgNr]++;
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
					}
				}
			});
		}
		lock([&]() -> void {
			// done with image
			map.erase(imgNr);

			for (int i = 0; i < nrOfEvaluations; i++) {
				swresult.evaluations["easy"][i].nrOfImagesEvaluated++;
				swresult.evaluations["moderate"][i].nrOfImagesEvaluated++;
				swresult.evaluations["hard"][i].nrOfImagesEvaluated++;
				swresult.combinedEvaluations[i].nrOfImagesEvaluated++;
			}
			nrOfImagesEvaluated++;
		});
	}, paralellization);

	// iteration with multiple threads is done, update the evaluation timings and the worst false positive/negatives

	for (int i = 0; i < nrOfEvaluations; i++) {
		swresult.evaluations["easy"][i].evaluationSpeedPerRegionMS = 1.0 * (featureBuildTime + sumTimesRegions) / nrRegions;
		swresult.evaluations["moderate"][i].evaluationSpeedPerRegionMS = 1.0 * (featureBuildTime + sumTimesRegions) / nrRegions;
		swresult.evaluations["hard"][i].evaluationSpeedPerRegionMS = 1.0 * (featureBuildTime + sumTimesRegions) / nrRegions;
		swresult.combinedEvaluations[i].evaluationSpeedPerRegionMS = 1.0 * (featureBuildTime + sumTimesRegions) / nrRegions;
	}

	ProgressWindow::getInstance()->finish(std::string(name));

	return swresult;
}