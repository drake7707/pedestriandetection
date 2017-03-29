#include "EvaluatorCascade.h"


EvaluatorCascade::EvaluatorCascade(std::string& name)
	: IEvaluator(name)
{
	resetClassifierHitCount();
}


EvaluatorCascade::~EvaluatorCascade()
{
}


double EvaluatorCascade::evaluateFeatures(FeatureVector& v) const {
	int classifierIndex;
	classifierHitCount[classifierIndex++];
	return evaluateCascadeFeatures(v, &classifierIndex);
}

double EvaluatorCascade::evaluateCascadeFeatures(FeatureVector& v, int* classifierIndex) const {

	double result;
	for (int i = 0; i < cascade.size(); i++)
	{
		*classifierIndex = i;
		result = cascade[i].model.evaluateFeatures(v);
		int resultClass = (result + cascade[i].valueShift) > 0 ? 1 : -1;

		if (resultClass == -1) {
			// stop cascade when a negative is encountered
			return result + cascade[i].valueShift;
		}
	}
	return result + cascade[cascade.size() - 1].valueShift;
}

void EvaluatorCascade::updateLastModelValueShift(double valueShift) {
	cascade[cascade.size() - 1].valueShift = valueShift;
}



void EvaluatorCascade::save(std::string& path) const {

	cv::FileStorage fs(path, cv::FileStorage::WRITE);

	fs << "trainingRound" << trainingRound;

	std::vector<std::string> names;
	for (auto& c : cascade)
		names.push_back(c.model.getName());

	std::vector<float> valueShifts;
	for (auto& c : cascade)
		valueShifts.push_back(c.valueShift);

	fs << "name_count" << (int)names.size();
	for (int i = 0; i < names.size(); i++)
		fs << ("name_" + std::to_string(i)) << names[i];

	fs << "valueShifts" << valueShifts;

	fs << "classifierHitCount" << classifierHitCount;


	fs.release();

}

void EvaluatorCascade::load(std::string& path, std::string& modelsDirectory) {
	if (!FileExists(path))
		throw std::exception("File does not exist");

	cv::FileStorage fsRead(path, cv::FileStorage::READ);


	std::vector<std::string> names;
	//fsRead["names"] >> names;
	int nameCount;
	fsRead["name_count"] >> nameCount;

	for (int i = 0; i < nameCount; i++)
	{
		std::string name;
		fsRead["name_" + std::to_string(i)] >> name;
		names.push_back(name);
	}


	std::vector<float> valueShifts;
	fsRead["valueShifts"] >> valueShifts;

	fsRead["trainingRound"] >> trainingRound;

	fsRead["classifierHitCount"] >> classifierHitCount;

	cascade.clear();
	for (int i = 0; i < names.size(); i++)
	{
		std::string name = std::string(names[i]);

		ModelEvaluator evaluator(name);
		evaluator.loadModel(modelsDirectory + PATH_SEPARATOR + name + ".xml");


		cascade.push_back(EvaluationCascadeEntry(evaluator, valueShifts[i]));
	}

	if (classifierHitCount.size() != cascade.size())
		resetClassifierHitCount();

	fsRead.release();
}