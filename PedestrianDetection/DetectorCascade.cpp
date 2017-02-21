#include "DetectorCascade.h"
#include "HistogramOfOrientedGradients.h"
#include <queue>




DetectorCascade::~DetectorCascade()
{
}


void DetectorCascade::iterateDataset(std::function<void(cv::Mat&)> tpFunc, std::function<void(cv::Mat&)> tnFunc, std::function<bool(int)> includeSample) {

	srand(7707);


	std::vector<DataSetLabel> labels = dataSet->getLabels();

	std::string currentNumber = "";
	std::vector<cv::Mat> currentImages;

	int idx = 0;
	for (auto& l : labels) {

		if (includeSample(idx)) {
			if (currentNumber != l.getNumber()) {
				currentNumber = l.getNumber();
				currentImages = dataSet->getImagesForNumber(currentNumber);
			}

			// get true positive and true negative image
			// -----------------------------------------

			cv::Mat rgbTP;
			cv::Rect2d& r = l.getBbox();
			if (r.x >= 0 && r.y >= 0 && r.x + r.width < currentImages[0].cols && r.y + r.height < currentImages[0].rows) {

				//for (auto& img : currentImages) {
				auto& img = currentImages[0];
				img(l.getBbox()).copyTo(rgbTP);
				cv::resize(rgbTP, rgbTP, cv::Size2d(refWidth, refHeight));

				// build training mat

				tpFunc(rgbTP);
				//truePositiveFeatures.push_back(resultTP.getFeatureArray());

				cv::Mat rgbTPFlip;
				cv::flip(rgbTP, rgbTPFlip, 1);

				tpFunc(rgbTPFlip);

				// take a number of true negative patches that don't overlap
				for (int k = 0; k < nrOfTN; k++)
				{
					// take an equivalent patch at random for a true negative
					cv::Mat rgbTN;
					cv::Rect2d rTN;

					int iteration = 0;
					do {
						double width = l.getBbox().width * (1 + rand() * 1.0 / RAND_MAX * sizeVariance);
						double height = l.getBbox().height * (1 + rand() * 1.0 / RAND_MAX * sizeVariance);
						rTN = cv::Rect2d(randBetween(0, img.cols - width), randBetween(0, img.rows - height), width, height);
					} while (iteration++ < 100 && ((rTN & l.getBbox()).area() > 0 || rTN.x < 0 || rTN.y < 0 || rTN.x + rTN.width >= img.cols || rTN.y + rTN.height >= img.rows));


					if (iteration < 100) {
						img(rTN).copyTo(rgbTN);
						cv::resize(rgbTN, rgbTN, cv::Size2d(refWidth, refHeight));
						tnFunc(rgbTN);
					}
				}
				//}
			}
		}
		idx++;
	}
}


void DetectorCascade::saveSVMLightFiles() {

	std::ofstream trainingFile("training.dat");
	if (!trainingFile.is_open())
		throw std::exception("Unable to create training file");

	iterateDataset([&](cv::Mat& mat) -> void {
		cv::Mat features = this->getFeatures(mat).toMat();

		trainingFile << "+1 ";
		for (int i = 0; i < features.cols; i++)
			trainingFile << (i + 1) << ":" << features.at<float>(0, i) << " ";
		trainingFile << "#";
		trainingFile << std::endl;
		
	}, [&](cv::Mat& mat) -> void {
		cv::Mat features = this->getFeatures(mat).toMat();

		trainingFile << "-1 ";
		for (int i = 0; i < features.cols; i++)
			trainingFile << (i + 1) << ":" << features.at<float>(0, i) << " ";
		trainingFile << "#";
		trainingFile << std::endl;
	}, [&](int idx) -> bool { return (idx % testSampleEvery != 0); });


	std::ofstream testFile("test.dat");
	if (!testFile.is_open())
		throw std::exception("Unable to create training file");

	iterateDataset([&](cv::Mat& mat) -> void {
		cv::Mat features = this->getFeatures(mat).toMat();

		testFile << "+1 ";
		for (int i = 0; i < features.cols; i++)
			trainingFile << (i + 1) << ":" << features.at<float>(0, i) << " ";
		testFile << "#";
		testFile << std::endl;

	}, [&](cv::Mat& mat) -> void {
		cv::Mat features = this->getFeatures(mat).toMat();
		testFile << "-1 ";
		for (int i = 0; i < features.cols; i++)
			trainingFile << (i + 1) << ":" << features.at<float>(0, i) << " ";
		testFile << "#";
		testFile << std::endl;
	}, [&](int idx) -> bool { return (idx % testSampleEvery == 0); });

}



void  DetectorCascade::getFeatureVectorsFromDataSet(std::vector<FeatureVector>& truePositiveFeatures, std::vector<FeatureVector>& trueNegativeFeatures) {

	iterateDataset([&](cv::Mat& mat) -> void {
		truePositiveFeatures.push_back(this->getFeatures(mat));
	}, [&](cv::Mat& mat) -> void {
		trueNegativeFeatures.push_back(this->getFeatures(mat));
	}, [&](int idx) -> bool { return idx % testSampleEvery != 0; });

}



void DetectorCascade::slideWindow(cv::Mat& img, std::function<void(cv::Rect2d bbox, cv::Mat& region)> func) {

	int slidingWindowWidth = 32;
	int slidingWindowHeight = 64;
	int slidingWindowStep = 8;
	int maxScaleReduction = 4; // 4 times reduction

	for (float invscale = 1; invscale <= maxScaleReduction; invscale += 1)
	{
		cv::Mat imgToTest;
		cv::resize(img, imgToTest, cv::Size2d(ceilTo(img.cols / invscale, slidingWindowWidth), ceilTo(img.rows / invscale, slidingWindowHeight)));

		for (int j = 0; j < imgToTest.rows / slidingWindowHeight - 1; j++)
		{
			for (float verticalStep = 0; verticalStep < slidingWindowWidth; verticalStep += slidingWindowStep / invscale)
			{
				for (int i = 0; i < imgToTest.cols / slidingWindowWidth - 1; i++)
				{

					for (float horizontalStep = 0; horizontalStep < slidingWindowWidth; horizontalStep += slidingWindowStep / invscale)
					{
						cv::Rect windowRect(i * slidingWindowWidth + horizontalStep, j * slidingWindowHeight + verticalStep, slidingWindowWidth, slidingWindowHeight);
						cv::Mat region;
						cv::resize(imgToTest(windowRect), region, cv::Size2d(refWidth, refHeight));

						func(windowRect, region);
					}
				}
			}
		}
	}
}


FeatureVector DetectorCascade::getFeatures(cv::Mat& mat) {
	auto result = getHistogramsOfOrientedGradient(mat, patchSize, binSize, false, true);
	FeatureVector vector = result.getFeatureArray(addS2);
	return vector;
}

double DetectorCascade::evaluateRegion(cv::Mat& region) {

	FeatureVector vec = getFeatures(region);

	double result;
	for (auto& d : cascade) {
		result = d.evaluate(vec);

		if (result <= 0) {
			// negative
			return result;
		}
		else {
			// positive
			// check next detector in the cascade
		}
	}
	return result;
}


std::vector<MatchRegion> DetectorCascade::evaluateImage(cv::Mat& img) {

	std::vector<MatchRegion> regions;
	slideWindow(img, [&](cv::Rect2d bbox, cv::Mat& region) -> void {


		double result = this->evaluateRegion(region);
		if (result > 0) {

			MatchRegion region;
			region.region = bbox;
			region.result = result;
			regions.push_back(region);
		}
	});
	return regions;
}

ClassifierEvaluation DetectorCascade::evaluateDetector(Detector& d, std::vector<FeatureVector>& truePositives, std::vector<FeatureVector>& trueNegatives) {
	ClassifierEvaluation evaluation;
	for (auto& tp : truePositives) {
		double result = d.evaluate(tp);
		if (result > 0)
			evaluation.nrOfTruePositives++;
		else
			evaluation.nrOfFalsePositives++;
	}

	for (auto& tn : trueNegatives) {
		double result = d.evaluate(tn);
		if (result > 0)
			evaluation.nrOfTrueNegatives++;
		else
			evaluation.nrOfFalseNegatives++;
	}
	return evaluation;
}

void DetectorCascade::buildCascade() {

	std::vector<FeatureVector> truePositives;
	std::vector<FeatureVector> trueNegatives;

	iterateDataset([&](cv::Mat& mat) -> void {
		FeatureVector vec = getFeatures(mat);
		truePositives.push_back(vec);

	}, [&](cv::Mat& mat) -> void {
		FeatureVector vec = getFeatures(mat);
		trueNegatives.push_back(vec);
	}, [&](int idx) -> bool { return  (idx % testSampleEvery != 0); });




	std::vector<FeatureVector> trainingTruePositives = truePositives;

	std::vector<FeatureVector> trainingTrueNegatives = trueNegatives;
	
	int iteration = 0;
	while (iteration++ < max_iterations) {
		std::cout << "Starting cascade building iteration " << iteration << std::endl;

		Detector d1;
		d1.buildModel(trainingTruePositives, trainingTrueNegatives);
		// adjust bias shift so d1 classifies 99% of TP correctly at the expense of TN
		// TODO
		d1.biasShift = 0.003;
		ClassifierEvaluation eval = evaluateDetector(d1, trainingTruePositives, trainingTrueNegatives);
		eval.print(std::cout);
		cascade.push_back(d1);

		// iterate over entire dataset, retrieving the most wrong negatives
		std::vector<FeatureVector> newTrueNegatives;
		auto comp = [](std::pair<FeatureVector, double> a, std::pair<FeatureVector, double> b) { return a.second > b.second; };
		std::priority_queue< std::pair<FeatureVector, double>, std::vector<std::pair<FeatureVector, double>>, decltype(comp) >
			queue(comp);

		std::vector<DataSetLabel> labels = dataSet->getLabels();

		std::map<std::string, std::vector<DataSetLabel>> labelsPerNumber;
		for (auto& l : labels)
			labelsPerNumber[l.getNumber()].push_back(l);


		for (auto& pair : labelsPerNumber) {
			std::vector<cv::Mat> currentImages = dataSet->getImagesForNumber(pair.first);

			std::cout << "Iterating dataset " << pair.first << std::endl;

			slideWindow(currentImages[0], [&](cv::Rect2d bbox, cv::Mat& region) -> void {

				double result = this->evaluateRegion(region);
				// check all labels and see if they don't overlap

				int realClass = -1;
				for (auto& l : pair.second) {

					double intersectionArea = (bbox & l.getBbox()).area();
					double unionArea = bbox.area() + l.getBbox().area() - intersectionArea;
					if (intersectionArea / unionArea > 0.5) {
						// it overlaps with a true positive
						realClass = 1;
						break;
					}
				}

				if (result < 0) {
					// negative
					if (realClass == 1) {
						// should have been positive
					}
				}
				else {
					if (realClass == -1) {
						// should have been negative
						queue.emplace(std::pair<FeatureVector, double>(this->getFeatures(region), result));
						if (queue.size() > truePositives.size()) {
							queue.pop();
						}

					}
				}
			});

			while (queue.size() > 0) {
				newTrueNegatives.push_back(queue.top().first);
				queue.pop();
			}

			// now we have a new training set to train the next detector on
			trainingTrueNegatives = newTrueNegatives;

		}
	}

}



void DetectorCascade::saveCascade(std::string& path) {
	for (int i = 0; i < cascade.size(); i++)
	{
		cascade[i].saveModel(path + PATH_SEPARATOR + "detector_" + std::to_string(i));
	}
}

void DetectorCascade::loadCascade(std::string& path) {
	for (int i = 0; i < max_iterations; i++)
	{
		cascade[i].loadModel(path + PATH_SEPARATOR + "detector_" + std::to_string(i));
	}
}

