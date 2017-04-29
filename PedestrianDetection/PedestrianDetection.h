#pragma once
#include "stdafx.h"

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ml.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>

#include "ProgressWindow.h"

#include "Helper.h"
#include "TrainingDataSet.h"

#include "HistogramOfOrientedGradients.h"
#include "LocalBinaryPatterns.h"

#include "ModelEvaluator.h"
#include "IFeatureCreator.h"
#include "HOGFeatureCreator.h"
#include "HOGHistogramVarianceFeatureCreator.h"
#include "CornerFeatureCreator.h"
#include "HistogramDepthFeatureCreator.h"
#include "SURFFeatureCreator.h"
#include "ORBFeatureCreator.h"
#include "SIFTFeatureCreator.h"
#include "CenSurEFeatureCreator.h"
#include "MSDFeatureCreator.h"
#include "BRISKFeatureCreator.h"
#include "FASTFeatureCreator.h"
#include "HDDFeatureCreator.h"
#include "LBPFeatureCreator.h"
#include "HONVFeatureCreator.h"
#include "CoOccurenceMatrixFeatureCreator.h"
#include "RAWRGBFeatureCreator.h"
#include "HOIFeatureCreator.h"
#include "SDDGFeatureCreator.h"
#include "RAWLUVFeatureCreator.h"


#include "EvaluatorCascade.h"

#include "ModelEvaluator.h"

#include "FeatureTester.h"

#include "FeatureSet.h"

#include "CoOccurenceMatrix.h"

#include "KITTIDataSet.h"
#include "KAISTDataSet.h"

#include "DataSet.h"

#include "JetHeatMap.h"
#include "EvaluationSettings.h"
#include "RiskAnalysis.h"


/// <summary>
/// Creates an initial training set for the given data set by selecting the true positives and random negatives that do not intersect with positives
/// </summary>
TrainingDataSet buildInitialTrainingSet(EvaluationSettings& settings, DataSet* dataSet);


void buildTesterJobsFromInputSets(FeatureTester& tester, DataSet* dataSet, EvaluationSettings& settings);

/// <summary>
/// Runs the sparse feature descriptors with multiple cluster sizes to determine the best one after the first round
/// </summary>
void evaluateClusterSize(FeatureTester& tester, EvaluationSettings settings);

/// <summary>
/// Checks the distance between true positives and true negatives to see if the distance decreases over multiple training rounds
/// </summary>
void checkDistanceBetweenTPAndTN(std::string& trainingFile, EvaluationSettings& settings, FeatureTester& tester, std::set<std::string> set, std::string& outputFile);


/// <summary>
/// Prints the average depth of true positives along with the height of the bounding box to a given file to correlate
/// </summary>
void printHeightVerticalAvgDepthRelation(std::string& trainingFile, EvaluationSettings& settings, std::ofstream& str);



/// <summary>
/// Tests the ROI selection on thermal images with given parameters
/// </summary>
void testKAISTROI(EvaluationSettings& settings, double w = 12, double alpha = 2, std::vector<float> scales = { 0.5, 0.75, 1 });


/// <summary>
/// Creates a mask based on the otsu threshold
/// </summary>
cv::Mat getMask(cv::Mat& roi);

/// <summary>
/// Tests a given classifier with the current settings and given value shift. All images of the data set will be evaluated and annotated with the result and saved to the working directory
/// </summary>
void testClassifier(FeatureTester& tester, EvaluationSettings& settings, std::string& dataset, std::set<std::string> set, double valueShift);

/// <summary>
/// Explains a given feature descriptor with a heat map
/// </summary>
void explainModel(FeatureTester& tester, EvaluationSettings& settings, std::set<std::string> set, std::string& dataset);

/// <summary>
/// Analyzes which positives will be missed with the sliding window approach with the given settings regardless of perfect classification or not
/// </summary>
void testSlidingWindow(EvaluationSettings& settings, std::string& dataset);

/// <summary>
/// Draws the risk categories on the dataset and shows a top down image
/// </summary>
void drawRiskOnDepthDataSet(DataSet* set);

/// <summary>
/// Dumps the bounding box heights per risk category to separate csv file
/// </summary>
void dumpHeightsPerRiskCategory(DataSet* set);

/// <summary>
/// Shows image per image in the given training set file and data set and annotates the negatives and positives
/// </summary>
void browseThroughTrainingSet(std::string& trainingFile, DataSet* dataSet);

/// <summary>
/// Creates a speed csv file that lists the speed of feature descriptors sets taken from the inputsets.txt
/// </summary>
void testSpeed(FeatureTester& tester, EvaluationSettings& settings);

/// <summary>
/// Prints the feature vector size for each feature descriptor
/// </summary>
void printFeatureVectorSize(FeatureTester& tester);

/// <summary>
/// Creates an average gradient of all the true positives in the dataset
/// </summary>
void createAverageGradient(EvaluationSettings& settings);

/// <summary>
/// Verifies the input set to test if the feature vectors with and without integral histograms are more or less the same
/// Multiple issues can cause them to deviate: borders are handled differently, floating point error accumulation, ...
/// </summary>
void verifyWithAndWithoutIntegralHistogramsFeaturesAreTheSame(FeatureTester& tester, EvaluationSettings& settings);

int main(int argc, char** argv);