#pragma once
#include <vector>
#include "IEvaluator.h"
#include "ModelEvaluator.h"



struct EvaluationCascadeEntry {

	EvaluationCascadeEntry(ModelEvaluator& model, double valueShift)
		: model(model), valueShift(valueShift) {

	}

	ModelEvaluator model;
	double valueShift;
};

class EvaluatorCascade : public IEvaluator
{

	std::vector<EvaluationCascadeEntry> cascade;

	std::vector<int> classifierHitCount;
	std::mutex lockClassifierHitCount;
	bool trackClassifierHitCount = false;

public:

	int trainingRound = 0;

	EvaluatorCascade(std::string& name);
	virtual ~EvaluatorCascade();

	/// <summary>
	/// Adds a model to the cascade with given decision boundary shift to apply to its result
	/// </summary>
	void addModelEvaluator(ModelEvaluator& model, double valueShift);

	/// <summary>
	/// Returns the models in the cascade
	/// </summary>
	std::vector<EvaluationCascadeEntry> getEntries() const;

	/// <summary>
	/// If enabled the track classifier hit count will track which classifiers are the decisive factor
	/// for rejecting or accepting the sample. Because this is only when the final evaluation is done and 
	/// it uses a lock to prevent multithreading issues it can be disabled
	/// </summary>
	void setTrackClassifierHitCountEnabled(bool enabled);

	/// <summary>
	/// Evaluates the given feature vector and returns the score and class (sign)
	/// </summary>
	virtual double evaluateFeatures(FeatureVector& v);

	/// <summary>
	/// Evaluates the given feature vector, but also returns which classifier was responsible for the result
	/// </summary>
	double evaluateCascadeFeatures(FeatureVector& v, int* classifierIndex);

	/// <summary>
	/// Resets the classifier hit count
	/// </summary>
	void resetClassifierHitCount();

	/// <summary>
	/// Returns the classifier hit count
	/// </summary>
	std::vector<int> getClassifierHitCount() const;

	/// <summary>
	/// Returns a model corresponding to given index
	/// </summary>
	ModelEvaluator getModelEvaluator(int idx) const;

	/// <summary>
	/// Updates the decision boundary shift of the last model. This is done when the classifier evaluation is complete
	/// and the TPR of 95% needs to be attained
	/// </summary>
	void updateLastModelValueShift(double valueShift);

	/// <summary>
	/// Saves the cascade to the given path
	/// </summary>
	void save(std::string& path) const;

	/// <summary>
	/// Clears and loads the cascade from the given path. The models directory is needed to load the models of the cascade as well
	/// </summary>
	void load(std::string& path, std::string& modelsDirectory);

	/// <summary>
	/// Returns the number of models in the cascade
	/// </summary>
	int size() const;
};

