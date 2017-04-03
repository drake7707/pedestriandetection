#include "SDDGFeatureCreator.h"






SDDGFeatureCreator::SDDGFeatureCreator(std::string& name, IFeatureCreator::Target target, int refWidth, int refHeight)
	: IFeatureCreator(name), refWidth(refWidth), refHeight(refHeight), target(target)
{
}


SDDGFeatureCreator::~SDDGFeatureCreator()
{
}


int SDDGFeatureCreator::getNumberOfFeatures() const {
	int nrOfCellsX = refWidth / cellSize;
	int nrOfCellsY = refHeight / cellSize;
	return nrOfCellsX * nrOfCellsY * SDDGLength;
}

FeatureVector SDDGFeatureCreator::getFeatures(cv::Mat& rgb, cv::Mat& depth, cv::Mat& thermal) const {
	int nrOfCellsX = refWidth / cellSize;
	int nrOfCellsY = refHeight / cellSize;

	cv::Mat img;
	if (target == IFeatureCreator::Target::RGB) {
		cv::cvtColor(rgb, img, CV_BGR2GRAY);
	}
	else if (target == IFeatureCreator::Target::Depth)
		img = depth;
	else
		img = thermal;

	cv::Mat gx, gy;
	cv::Sobel(img, gx, CV_32F, 1, 0, 1);
	cv::Sobel(img, gy, CV_32F, 0, 1, 1);
	cv::Mat magnitude(img.rows, img.cols, CV_32FC1, cv::Scalar(0));


	FeatureVector fVector;
	fVector.reserve(getNumberOfFeatures());

	for (int j = 0; j < img.rows; j++)
	{
		for (int i = 0; i < img.cols; i++)
		{
			float gradX = gx.at<float>(j, i);
			float gradY = gy.at<float>(j, i);

			magnitude.at<float>(j, i) = sqrt(gradX * gradX + gradY * gradY);
		}
	}

	for (int j = 0; j < nrOfCellsY; j++)
	{
		for (int i = 0; i < nrOfCellsX; i++)
		{
			std::vector<float> SDDG = calculateSDDG(cellSize, cellSize, [&](int x, int y) -> float {
				float mag = magnitude.at<float>(y + cellSize * j, x + cellSize * i);
				return mag;
			});

			for (int s = 0; s < SDDG.size(); s++)
				fVector.push_back(SDDG[s]);
		}
	}

	return fVector;
}

cv::Mat SDDGFeatureCreator::explainFeatures(int offset, std::vector<float>& weightPerFeature, std::vector<float>& occurrencePerFeature, int refWidth, int refHeight) const {
	cv::Mat explanation;

	int nrOfCellsX = refWidth / cellSize;
	int nrOfCellsY = refHeight / cellSize;

	int idx = 0;
	for (int j = 0; j < nrOfCellsY; j++)
	{
		for (int i = 0; i < nrOfCellsX; i++)
		{

			float weight = 0;
			for (int k = 0; k < SDDGLength; k++)
			{
				weight += occurrencePerFeature[offset + idx];
				idx++;
			}
			int offsetX = i * cellSize;
			int offsetY = j * cellSize;
			cv::rectangle(explanation, cv::Rect(offsetX, offsetY, cellSize, cellSize), cv::Scalar(weight), -1);
		}
	}

	return explanation;
}

std::vector<bool> SDDGFeatureCreator::getRequirements() const {
	return{ target == IFeatureCreator::Target::RGB,
		target == IFeatureCreator::Target::Depth,
		target == IFeatureCreator::Target::Thermal
	};
}



void SDDGFeatureCreator::bresenhamLine(int srcX, int srcY, int dstX, int dstY, std::function<void(int x, int y)> setPixelFunc) const {
	// bresenham's line algorithm
	int x = srcX;
	int y = srcY;
	int x2 = dstX;
	int y2 = dstY;

	int w = x2 - x;
	int h = y2 - y;
	int dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;
	if (w < 0) dx1 = -1; else if (w > 0) dx1 = 1;
	if (h < 0) dy1 = -1; else if (h > 0) dy1 = 1;
	if (w < 0) dx2 = -1; else if (w > 0) dx2 = 1;
	int longest = abs(w);
	int shortest = abs(h);
	if (!(longest > shortest)) {
		longest = abs(h);
		shortest = abs(w);
		if (h < 0) dy2 = -1; else if (h > 0) dy2 = 1;
		dx2 = 0;
	}
	int numerator = longest >> 1;
	for (int i = 0; i <= longest; i++) {

		setPixelFunc(x, y);

		numerator += shortest;
		if (!(numerator < longest)) {
			numerator -= longest;
			x += dx1;
			y += dy1;
		}
		else {
			x += dx2;
			y += dy2;
		}
	}
}

std::vector<float> SDDGFeatureCreator::calculateSDDG(int cellWidth, int cellHeight, std::function<float(int x, int y)> gradientAt) const {


	// Symmetrical layered gradient difference (SLGD)
	std::vector<std::vector<float>> SLGD;

	// Symmetrical average gradient difference (SAGD)
	std::vector<float> SAGD;

	// Symmetrical hyperplane gradient difference (SHGD)
	std::vector<float> SHGD;

	// Final Scattered Difference of Directional Gradients (SDDG)
	std::vector<float> SDDG;

	for (int hyperplane = 0; hyperplane < 8; hyperplane++) {

		std::vector<float> SLGD_i = std::vector<float>();
		int sumM_ij = 0; // sum of all pixels pairs in all layers
		float sumDiff = 0; // sum of all differences of all layers

		int x0, y0, x1, y1;

		if (hyperplane >= 0 && hyperplane <= 4) {
			x0 = 0;
			y0 = hyperplane * 2;
			x1 = cellWidth - 1;
			y1 = cellHeight - 1 - y0;
		}
		else if (hyperplane > 4 && hyperplane < 8) {
			x0 = (hyperplane - 4) * 2;
			y0 = cellHeight - 1;
			x1 = cellWidth - 1 - x0;
			y1 = 0;
		}

		int dy = y1 - y0;
		int dx = x1 - x0;
		// go over each layer on distance away from the hyperplane
		for (int distance = 1; distance < cellWidth; distance++) {
			int M_ij = 0; // sum of pixel pairs in layer on distance away from the hyperplane
			float sumAbsDiff = 0;

			bresenhamLine(x0, y0, x1, y1, [&](int x, int y) -> void {
				int posX = 0; int posY = 0;
				int negX = 0; int negY = 0;

				if (abs(dy) == abs(dx)) {
					if (dy / dx > 0) {
						posX = x - distance;
						posY = y;
						negX = x;
						negY = y - distance;
					}
					else {
						posX = x;
						posY = y + distance;
						negX = x - distance;
						negY = y;
					}
				}
				else if (abs(dy) > abs(dx)) {
					posX = x + distance;
					posY = y;
					negX = x - distance;
					negY = y;
				}
				else {
					posX = x;
					posY = y + distance;
					negX = x;
					negY = y - distance;
				}


				if (posX >= 0 && posX < cellWidth &&
					posY >= 0 && posY < cellHeight &&
					negX >= 0 && negX < cellWidth &&
					negY >= 0 && negY < cellHeight) {

					float diff = gradientAt(posX, posY) - gradientAt(negX, negY);
					sumDiff += diff;
					sumAbsDiff += abs(diff);
					M_ij++;
				}
			});

			if (M_ij > 0) {
				float avg = sumAbsDiff / M_ij;
				SLGD_i.push_back(avg);
			}

			sumM_ij += M_ij;
		}

		SLGD.push_back(SLGD_i);

		float SAGD_i = abs(sumDiff) / sumM_ij;
		SAGD.push_back(SAGD_i);


		// now iterate on the pixels on the hyperplane itself 
		// starting from the center away
		std::vector<cv::Point> points;
		bresenhamLine(x0, y0, x1, y1, [&](int x, int y) -> void {
			points.push_back(cv::Point(x, y));
		});

		sumDiff = 0;
		int halfM_i0 = 0;
		int middle = points.size() / 2;
		for (int i = 0; i < points.size() / 2; i++) {
			auto& pos = points[middle + i];
			auto& neg = points[middle - i];

			float diff = gradientAt(pos.x, pos.y) - gradientAt(neg.x, neg.y);
			sumDiff += diff;
			halfM_i0++;
		}
		float SHGD_i = abs(sumDiff) / halfM_i0;
		SHGD.push_back(SHGD_i);
	}


	for (int hyperplane = 0; hyperplane < 8; hyperplane++) {
		// SLGD
		for (int i = 0; i < SLGD[hyperplane].size(); i++) {
			SDDG.push_back(SLGD[hyperplane][i]);
		}
		// SAGD
		SDDG.push_back(SAGD[hyperplane]);

		// SHGD
		SDDG.push_back(SHGD[hyperplane]);
	}

	return SDDG;
}

