#include "adaboost.hpp"
#include "weak_classifier.hpp"
#include "test.h"
using namespace DM_AG;
//
//int mainAdaBoostTest() {
//
//	Classifier<int>::Data data;
//	Labels labels;
//
//	const unsigned int number_features = 200;
//
//	// SIMULATE A TRAINER
//
//	std::cout << "Training ... " << std::endl;
//
//	for (unsigned int i = 0; i < number_features; i++) {
//
//		data.push_back(i);
//		// ground-truth classifier (TRAINER)
//
//		(i > 70 && i < 130) ?
//			labels.push_back(1) :  // good
//			labels.push_back(-1);  // bad
//	}
//	std::cout << "Done ... " << std::endl;
//	std::cout << "WeakClassifier ... " << std::endl;
//	//
//	// Create the pool of classifiers
//	//
//	Classifier<int>::CollectionClassifiers classifiers;
//	classifiers.push_back(new WeakClassifierOne());
//	classifiers.push_back(new WeakClassifierTwo());
//	classifiers.push_back(new WeakClassifierThree());
//	classifiers.push_back(new WeakClassifierFour());
//
//	//
//	// Ada boosting
//	//
//	ADA<int> ada;
//	std::cout << "Boosting ... " << std::endl;
//	ClassificationResults weights =
//		ada.ada_boost(classifiers, data, labels, 100);
//	std::cout << "Done " << std::endl;
//
//
//	unsigned int classifiers_size = classifiers.size();
//	for (unsigned int i = 0; i < classifiers_size; i++) {
//		std::cout << " w=" << weights[i];
//	}
//	std::cout << std::endl;
//
//	// 
//	// Start the strong classifier
//	//
//	StrongClassifier<int> sc(weights, &classifiers, labels);
//
//	for (unsigned int i = 0; i < number_features; i++) {
//		std::cout << " " << i << "," << sc.analyze(i);
//	}
//
//	getchar();
//	return 0;
//}
