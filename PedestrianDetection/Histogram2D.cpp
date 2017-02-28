#include "Histogram2D.h"

Histogram2D::Histogram2D() {

}

Histogram2D::Histogram2D(int size, float defaultValue) : std::vector<std::vector<float>>(size, std::vector<float>(size, defaultValue)) {

}

Histogram Histogram2D::flatten() const {
	Histogram dst(this->size() * this->size(), 0);

	int idx = 0;
	for (int j = 0; j < this->size(); j++)
	{
		for (int i = 0; i < this->size(); i++)
		{
			dst[idx] = at(j).at(i);
			idx++;
		}
	}
	return dst;
}