#include "MarchingSquares.h"

namespace Algorithms {

	std::vector<cv::Point> getUV(int nr, float threshold, float vLT, float vLB, float vRT, float vRB) {

		// (1 - min) / (max-min)
		float lInterpol = (threshold - vLT) / (vLB - vLT);
		float rInterpol = (threshold - vRT) / (vRB - vRT);

		float tInterpol = (threshold - vLT) / (vRT - vLT);
		float bInterpol = (threshold - vLB) / (vRB - vLB);

		cv::Point l = cv::Point(0, lInterpol);
		cv::Point t = cv::Point(tInterpol, 0);
		cv::Point r = cv::Point(1, rInterpol);
		cv::Point b = cv::Point(bInterpol, 1);

		std::vector<cv::Point> pts;
		switch (nr) {
		case 0:
			break;
		case 1: pts = { l, t };		break;
		case 2: pts = { t, r };		break;
		case 3: pts = { l, r };		break;

		case 4: pts = { r, b };		break;
		case 5: pts = { b, l };		break;
		case 6: pts = { b, t };		break;
		case 7: pts = { b, l };		break;

		case 8: pts = { b, l };		break;
		case 9: pts = { b, t };		break;
		case 10: pts = { b, r };	break;
		case 11: pts = { b, r };	break;

		case 12: pts = { l, r };	break;
		case 13: pts = { t, r };	break;
		case 14: pts = { t, l };	break;
		case 15:
			break;

		}
		return pts;
	}

	void marchingSquares(std::function<float(int, int)> func, std::function<void(int x1, int y1, int x2, int y2)> lineFunc, float targetValue, int imgWidth, int imgHeight, int nrCellsX, int nrCellsY) {
		std::vector<bool> mask;

		float cellSizeWidth = 1.0 * imgWidth / nrCellsX;
		float cellSizeHeight = 1.0 * imgHeight / nrCellsY;

		for (int j = 0; j < nrCellsY; j++) {
			for (int i = 0; i < nrCellsX; i++) {


				mask = {
					(j + 1) * cellSizeHeight + cellSizeHeight / 2 >= imgHeight ? false : func((j + 1) * cellSizeHeight + cellSizeHeight / 2,i * cellSizeWidth + cellSizeWidth / 2) > targetValue,
					(i + 1) * cellSizeWidth + cellSizeWidth / 2 >= imgWidth || (j + 1) * cellSizeHeight + cellSizeHeight / 2 >= imgHeight ? false : func((j + 1) * cellSizeHeight + cellSizeHeight / 2,(i + 1) * cellSizeWidth + cellSizeWidth / 2) > targetValue,
					(i + 1) * cellSizeWidth + cellSizeWidth / 2 >= imgWidth ? false : func(j * cellSizeHeight + cellSizeHeight / 2,(i + 1) * cellSizeWidth + cellSizeWidth / 2) > targetValue,
					func(j * cellSizeHeight + cellSizeHeight / 2,i * cellSizeWidth + cellSizeWidth / 2) > targetValue,
				};
				int nr = ((mask[0] ? 1 : 0) << 3) +
					((mask[1] ? 1 : 0) << 2) +
					((mask[2] ? 1 : 0) << 1) +
					((mask[3] ? 1 : 0) << 0);

				if (nr != 0 && nr != 15) {
					std::vector<cv::Point> uv = getUV(nr, targetValue,
						func(j * cellSizeHeight + cellSizeHeight / 2, i * cellSizeWidth + cellSizeWidth / 2),
						(j + 1) * cellSizeHeight + cellSizeHeight / 2 >= imgHeight ? targetValue : func((j + 1) * cellSizeHeight + cellSizeHeight / 2, i * cellSizeWidth + cellSizeWidth / 2),
						(i + 1) * cellSizeWidth + cellSizeWidth / 2 >= imgWidth ? targetValue : func(j * cellSizeHeight + cellSizeHeight / 2, (i + 1) * cellSizeWidth + cellSizeWidth / 2),
						(i + 1) * cellSizeWidth + cellSizeWidth / 2 >= imgWidth || (j + 1) * cellSizeHeight + cellSizeHeight / 2 >= imgHeight ? targetValue : func((j + 1) * cellSizeHeight + cellSizeHeight / 2, (i + 1) * cellSizeWidth + cellSizeWidth / 2)
					);

					if (uv.size() > 0) {
						float x = i * imgWidth / nrCellsX;
						float y = j * imgHeight / nrCellsY;


						float x1 = x + 1.0 * cellSizeWidth * uv[0].x + cellSizeWidth / 2;
						float y1 = y + 1.0 * cellSizeHeight *uv[0].y + cellSizeHeight / 2;
						float x2 = x + 1.0 * cellSizeWidth * uv[1].x + cellSizeWidth / 2;
						float y2 = y + 1.0 * cellSizeHeight *uv[1].y + cellSizeHeight / 2;
						lineFunc(x1, y1, x2, y2);
					}
				}
			}
		}
	}

};