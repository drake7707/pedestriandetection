#include "VariableNumberFeatureCreator.h"
#include "Helper.h"
#include "ProgressWindow.h"


VariableNumberFeatureCreator::VariableNumberFeatureCreator(std::string& creatorName, int clusterSize)
	: IFeatureCreator(creatorName), clusterSize(clusterSize)
{

	// try and load first
	std::string featureCachePath = (std::string("featurecache") + PATH_SEPARATOR + getName() + ".xml");
	if (FileExists(featureCachePath))
		loadCentroids(featureCachePath);
}


VariableNumberFeatureCreator::~VariableNumberFeatureCreator()
{
}

void VariableNumberFeatureCreator::prepare(TrainingDataSet& trainingDataSet) {

	std::string name = "Feature " + this->getName() + " preparation";
	std::string featureCachePath = (std::string("featurecache") + PATH_SEPARATOR + getName() + ".xml");

	if (FileExists(featureCachePath))
		return;

	int k = clusterSize;
	cv::TermCriteria criteria(CV_TERMCRIT_ITER, 1000, 1.0);
	int attempts = 10;
	int flags = cv::KmeansFlags::KMEANS_PP_CENTERS; // use kmeans++

	std::vector<FeatureVector> samples;

	trainingDataSet.iterateDataSet([](int idx) -> bool { return true; },
		[&](int idx, int resultClass, int imageNumber, cv::Rect region, cv::Mat&rgb, cv::Mat&depth) -> void {

		if (idx % 100 == 0)
			ProgressWindow::getInstance()->updateStatus(name, 1.0 * imageNumber / trainingDataSet.getNumberOfImages(), std::string("Aggregating feature vectors (") + std::to_string(imageNumber) + ")");

		// aggregate vectors

		std::vector<FeatureVector> localSamples = getVariableNumberFeatures(rgb, depth);
		for (auto& s : localSamples)
			samples.push_back(s);
	});

	if (samples.size() > 0) {
		int featureSize = samples[0].size();
		cv::Mat kmeansData(samples.size(), featureSize, CV_32FC1);

		for (int s = 0; s < samples.size(); s++)
		{
			for (int f = 0; f < featureSize; f++)
				kmeansData.at<float>(s, f) = samples[s][f];
		}


		ProgressWindow::getInstance()->updateStatus(name, 0, std::string("Running k-means clustering"));

		cv::Mat labels;
		cv::Mat centers;
		double result = cv::kmeans(kmeansData, k, labels, criteria, attempts, flags, centers);

		for (int j = 0; j < centers.rows; j++)
		{
			FeatureVector centroid(centers.cols, 0);
			for (int i = 0; i < centers.cols; i++)
			{
				centroid[i] = centers.at<float>(j, i);
			}
			this->centroids.push_back(centroid);
		}

		ProgressWindow::getInstance()->finish(name);

		//	// visualize to check if it's correct:
		//	cv::Mat test(128, 64, CV_8UC3, cv::Scalar(0));

		//	std::vector<cv::Scalar> colors;
		//	for (int i = 0; i < centroids.size(); i++)
		//	{
		//		colors.push_back(cv::Scalar(1.0 * rand() / RAND_MAX * 255, 1.0 * rand() / RAND_MAX * 255, 1.0 * rand() / RAND_MAX * 255));
		//	}

		//	for (int j = 0; j < samples.size(); j++)
		//	{
		//		float x = samples[j][0];
		//		float y = samples[j][1];

		//		double minDistance = std::numeric_limits<double>().max();
		//		int closestCentroidIndex = -1;
		//		for (int c = 0; c < centroids.size(); c++)
		//		{
		//			double distance = samples[j].distanceToSquared(centroids[c]);
		//			if (minDistance > distance) {
		//				closestCentroidIndex = c;
		//				minDistance = distance;
		//			}
		//		}
		//		cv::circle(test, cv::Point(x, y), 2, colors[closestCentroidIndex], -1);
		//	}

		//	for (int c = 0; c < centroids.size(); c++)
		//	{
		//		cv::circle(test, cv::Point(centroids[c][0], centroids[c][1]), 4, cv::Scalar(255, 0, 0), -1);
		//	}
		//	cv::imshow("test", test);
		//	cv::waitKey(0);
	}

	saveCentroids(featureCachePath);
}

FeatureVector VariableNumberFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth) const {
	std::vector<FeatureVector> localSamples = getVariableNumberFeatures(rgb, depth);


	// tally closest centroids
	FeatureVector histogram(centroids.size(), 0);
	for (auto& s : localSamples) {

		// find the centroid that's closest
		double minDistance = std::numeric_limits<double>().max();
		int closestCentroidIndex = -1;
		for (int c = 0; c < centroids.size(); c++)
		{
			double distance = s.distanceToSquared(centroids[c]);
			if (minDistance > distance) {
				closestCentroidIndex = c;
				minDistance = distance;
			}
		}

		histogram[closestCentroidIndex]++;
	}

	// because the amount of samples can vary the histogram must be normalized so the values can be compared
	if (localSamples.size() > 0) {
		for (int i = 0; i < histogram.size(); i++)
			histogram[i] /= localSamples.size();
	}
	return histogram;
}

int VariableNumberFeatureCreator::getNumberOfFeatures() const {
	return centroids.size();
}


cv::Mat VariableNumberFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	// TODO draw centroids as keypoints
	cv::Mat explanation(cv::Size(refWidth, refHeight), CV_32FC1, cv::Scalar(0));

	for (int i = 0; i < centroids.size(); i++) {

		double weight = occurrencePerFeature[offset + i];

		cv::KeyPoint p;
		p.pt = cv::Point2f(centroids[i][0], centroids[i][1]);
		p.size = centroids[i][2];
		p.octave = centroids[i][3];
		p.angle = centroids[i][4];
		p.class_id = centroids[i][5];
		std::vector<cv::KeyPoint> keypoints = { p };
		cv::Scalar color(weight);
		cv::Mat outputImg;

		int radius = p.size / 2;
		cv::circle(explanation, p.pt, radius, color, 1);
		cv::line(explanation, p.pt, cv::Point(p.pt.x + radius * cos(p.angle / 180 * CV_PI), p.pt.y + radius * sin(p.angle / 180 * CV_PI)), color, 1);

		//cv::
		//cv::drawKeypoints(explanation, keypoints, explanation, color);// , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	}
	return explanation;
}


void VariableNumberFeatureCreator::saveCentroids(std::string& path) {

	cv::FileStorage fs(path, cv::FileStorage::WRITE);
	fs << "centroids" << centroids;
	fs.release();
}

void VariableNumberFeatureCreator::loadCentroids(std::string& path) {

	cv::FileStorage fsRead(path, cv::FileStorage::READ);
	fsRead["centroids"] >> centroids;
	fsRead.release();
}


