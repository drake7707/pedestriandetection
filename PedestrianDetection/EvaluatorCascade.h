#pragma once
#include <vector>
#include "IEvaluator.h"
#include "ModelEvaluator.h"



struct EvaluationCascadeEntry {

	EvaluationCascadeEntry(ModelEvaluator& model, double valueShift)
		:  model(model), valueShift(valueShift) {

	}

	ModelEvaluator model;
	double valueShift;
};

class EvaluatorCascade : public IEvaluator
{

	std::vector<EvaluationCascadeEntry> cascade;

	

public:

	int trainingRound = 0;

	EvaluatorCascade(std::string& name);
	virtual ~EvaluatorCascade();

	void addModelEvaluator(ModelEvaluator& model, double valueShift) {
		cascade.push_back(EvaluationCascadeEntry(model, valueShift));
	}

	std::vector<EvaluationCascadeEntry> getEntries() const {
		return cascade;
	}

	virtual double evaluateFeatures(FeatureVector& v) const;

	void updateLastModelValueShift(double valueShift);

	void save(std::string& path) const;
	void load(std::string& path, std::string& modelsDirectory);
};

