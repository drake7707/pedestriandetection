#ifndef ADA_BOOST
#define ADA_BOOST 1
#include <vector>
//#include <boost/ptr_container/ptr_vector.hpp>
//#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include "opencv2\opencv.hpp"

/*

Source code from: http://codingplayground.blogspot.be/2009/03/adaboost-improve-your-performance.html

*/


struct Point2D {
	float x;
	float y;
	Point2D(float x, float y) : x(x), y(y) { }

	bool operator<(const Point2D& p) const {
		if (x < p.x)
			return true;
		else if (x > p.x)
			return false;
		else
		{
			if (y < p.y)
				return true;
			else
				return false;
		}
	}

	bool operator==(const Point2D& p) const {
		return p.x == x && p.y == y;
	}

private:
	friend std::ostream & operator<<(std::ostream &os, const Point2D& p);
};


namespace DM_AG {

	typedef std::vector<float> ClassificationResults;
	typedef cv::Mat MatrixResults;
	typedef std::vector<int> Labels;

	// A classifier
	//    abstract class

	template <typename T>
	class Classifier {

	public:
		typedef typename std::vector<Classifier<T>*> CollectionClassifiers;
		typedef typename std::vector<T> Data;

		virtual int analyze(const T& feature) const = 0;
	};

	template <typename T>
	class ADA {

	public:

		// 
		// Apply Adaboost 
		//
		//  @param weak_classifiers, a set of weak classifiers
		//  @param data, the dataset to classify
		//  @param labels, classification labels (e.g. -1; +1}
		//  @param num_rounds, # boost iteration (default 100)

		ClassificationResults
			ada_boost(typename Classifier<T>::CollectionClassifiers const &weak_classifiers,
				typename Classifier<T>::Data const & data,
				const Labels & labels,
				const unsigned int num_iterations) {

			// following notation
			//
			// http://en.wikipedia.org/wiki/AdaBoost

			ClassificationResults alpha;
			ClassificationResults D;

			size_t labels_size = labels.size();
			size_t classifiers_size = weak_classifiers.size();

			D.resize(labels_size);           // D
			alpha.resize(classifiers_size);       // alpha   

			// init result matrix for weak classifiers

			MatrixResults weak_classifiers_results(classifiers_size, labels_size, CV_32FC1, cv::Scalar(0));
			// Run each weak classifer
			//
			unsigned int num_current_classifier = 0;
			typename Classifier<T>::CollectionClassifiers::const_iterator wc =
				weak_classifiers.begin();
			typename Classifier<T>::CollectionClassifiers::const_iterator wc_end =
				weak_classifiers.end();
			for (; wc != wc_end; ++wc) {

				//std::cout << "\tClassifier=" << num_current_classifier << std::endl;

				for (unsigned int j = 0; j < labels_size; j++) {

					// store the result for feature i
					weak_classifiers_results.at<float>(num_current_classifier, j) = (*wc)->analyze(data[j]);

					//	std::cout << " " << j << "," << weak_classifiers_results.at<float>(num_current_classifier, j);
				}
				//std::cout << std::endl;
				num_current_classifier++;
			}

			// Init boosters
			for (unsigned int j = 0; j < labels_size; j++)
				D[j] = (1.0) / labels_size;   // init D

				  // for the maximum rounds
				  //
			for (unsigned int round = 0; round < num_iterations; round++) {

				std::cout << "Iteration" << round << std::endl;
				for (int i = 0; i < classifiers_size; i++)
				{
					std::cout << " " << i << " : " << alpha[i] << std::endl;
				}
				std::cout << "Sample weights" << std::endl;

				cv::Mat weightMat(200, 200, CV_8UC1, cv::Scalar(0));

				for (int i = 0; i < labels_size; i++)
				{
			/*		std::string type;
					if (deltaWeights[i] > 0)
						type = "+";
					else if (deltaWeights[i] < 0)
						type = "-";
					else
						type = "=";
					if (type != "=")
						std::cout << " " << i << " : " << D[i] << " (" << type << ") " << data[i] << " result of classifier: " << weak_classifiers_results.at<float>(best_classifier, i) << ", actual label: " << labels[i] << std::endl;*/

					auto p = (Point2D)data[i];
					cv::circle(weightMat, cv::Point((p.x * 200), 200-(p.y * 200)), 3, cv::Scalar(D[i] * 255), -1);

				}
				cv::normalize(weightMat, weightMat, 0, 255, cv::NormTypes::NORM_MINMAX);
				imshow("Weights at round " + std::to_string(round), weightMat);


				float min_error = labels_size;
				unsigned int best_classifier = 0;
				//
				// for each classifier
				for (num_current_classifier = 0; num_current_classifier < classifiers_size; num_current_classifier++) {

					float error = 0;
					// 
					// for each feature
					for (unsigned int j = 0; j < labels_size; j++)
						if (weak_classifiers_results.at<float>(num_current_classifier, j) != labels[j])
							error += D[j];

					//	std::cout << "Error for classifier " << num_current_classifier << " : " << std::fixed << error << std::endl;
					if (error < min_error) {
						min_error = error; // this is the best observed 
						best_classifier = num_current_classifier;
					}
				}// each classifier

				std::cout << "\tbest_classifier=" << best_classifier << " error=" << min_error << std::endl;

				if (min_error >= 0.5)    // GOOD enough 
					break;                 // condition  

				  // a_t 
				alpha[best_classifier] = log((1.0f + min_error) / (1.0f - min_error)) / 2;

				// D_{t+1}
				ClassificationResults D_1(D);


				std::vector<float> deltaWeights;

				// update D_{t+1}
				float z = 0;
				for (unsigned int j = 0; j < labels_size; j++) {

					D_1[j] = D[j] *
						exp(-alpha[best_classifier] *
							labels[j] *
							weak_classifiers_results.at<float>(best_classifier, j));
					z += D_1[j];
				}

				// normalize so that it is a prob distribution
				for (unsigned int j = 0; j < labels_size; j++) {
					deltaWeights.push_back(D_1[j] / z - D[j]);
					D[j] = D_1[j] / z;
				}
			


			} // all the rounds.

			return alpha;
		};

	}; // class ADA

	//
	// A strong classifier is a linear comb of weak class
	//
	template <typename T>
	class StrongClassifier : public Classifier<T>
	{
	private:
		ClassificationResults weigths_;
		size_t num_classifiers_;
		typename Classifier<T>::CollectionClassifiers * classifiers_;
		Labels & labels_;

	public:
		StrongClassifier(ClassificationResults & w,
			typename Classifier<T>::CollectionClassifiers * cls,
			Labels & labels) :
			weigths_(w),
			num_classifiers_(cls->size()),
			classifiers_(cls),
			labels_(labels) {};

		// analyze
		//
		int analyze(const T & feature) const {

			float val = 0;
			for (unsigned int current_classifier = 0; current_classifier < num_classifiers_; current_classifier++)
				val += weigths_[current_classifier] * (*classifiers_)[current_classifier]->analyze(feature);

			if (val >= 0)
				return 1;  // label +1
			else
				return -1; // label -1 
		};


	}; // strong classifier

} // namespace

#endif
