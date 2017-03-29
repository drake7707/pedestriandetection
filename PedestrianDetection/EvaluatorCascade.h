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

	virtual double evaluateFeatures(FeatureVector& v) const;
	double evaluateCascadeFeatures(FeatureVector& v, int* classifierIndex) const;


	void resetClassifierHitCount() {
		classifierHitCount = std::vector<int>(cascade.size(), 0);
	}

	void updateLastModelValueShift(double valueShift);

	void save(std::string& path) const;
	void load(std::string& path, std::string& modelsDirectory);


	int size() const {
		return cascade.size();
	}
};

