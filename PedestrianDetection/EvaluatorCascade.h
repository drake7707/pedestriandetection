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

	void addModelEvaluator(ModelEvaluator& model, double valueShift) {
		cascade.push_back(EvaluationCascadeEntry(model, valueShift));
		resetClassifierHitCount();
	}

	std::vector<EvaluationCascadeEntry> getEntries() const {
		return cascade;
	}

	void setTrackClassifierHitCountEnabled(bool enabled) {
		trackClassifierHitCount = enabled;
	}

	virtual double evaluateFeatures(FeatureVector& v);
	double evaluateCascadeFeatures(FeatureVector& v, int* classifierIndex);


	void resetClassifierHitCount() {
		classifierHitCount = std::vector<int>(cascade.size(), 0);
	}

	std::vector<int> getClassifierHitCount() const {
		return classifierHitCount;
	}

	ModelEvaluator getModelEvaluator(int idx) const {
		return cascade[idx].model;
	}

	void updateLastModelValueShift(double valueShift);

	void save(std::string& path) const;
	void load(std::string& path, std::string& modelsDirectory);


	int size() const {
		return cascade.size();
	}
};

