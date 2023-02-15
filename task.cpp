#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <chrono>
#include <climits>
#include <thread>
#include <algorithm>
#include <memory>
#include <immintrin.h>

static inline __m512 mult_and_sum_zfvalues(__m512 a, __m512 b, __m512 c) { // a * b + c
	return _mm512_fmadd_ps(a, b, c);
}

static inline __m512 sum_zfvalues(__m512 a, __m512 b) {
	return _mm512_add_ps(a, b);
}

static inline __m512 subtract_zfvalues(__m512 a, __m512 b) {
	return _mm512_sub_ps(a, b);
}

static inline __m512 mult_zfvalues(__m512 a, __m512 b) {
	return _mm512_mul_ps(a, b);
}

static inline __m512 set_zfvalue(float f) { // broadcast
	return _mm512_set1_ps(f);
}

static inline void store_zfvalue_unaligned(void* addr, __m512 val) {
	_mm512_storeu_ps(addr, val);
}

static inline __m512 load_zfvalue_unaligned(const void* addr) {
	return _mm512_loadu_ps(addr);
}

static inline __mmask16 cmpr_less_then_zfvalues(__m512 a, __m512 b) {
	return _mm512_cmplt_ps_mask(a, b);
}

class SimpleTimer 
{
public:
	SimpleTimer() {
		start = std::chrono::high_resolution_clock::now();
	}
	~SimpleTimer() {
		end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float> duration = end - start;
		std::cout << duration.count() << std::endl;
	}
private:
	std::chrono::time_point<std::chrono::steady_clock> start, end;
};

struct vec2
{
	float x = 0.0f;
	float y = 0.0f;
};

struct unit
{
	vec2 position; // position of unit (-10^5...10^5, -10^5...10^5)
	vec2 direction; // normalized view direction
	float fov_deg = 0.0f; // horizontal field-of-view in degrees (0...180)
	float distance = 0.0f; // view distance (0...10^5)
};

int number_of_bits(uint16_t x) {
	int res = 0;
	while (x) {
		++res;
		x &= x - 1;
	}
	return res;
}

int isInSector(const vec2& sec_end, const vec2& sec_begin, const float dist,
			   const std::vector<float>& x_coords, const std::vector<float>& y_coords) {
	int n = x_coords.size();
	int mainsz = (n / 16) * 16;
	__m512 sec_end_y		= set_zfvalue(-sec_end.y);
	__m512 sec_end_x		= set_zfvalue(sec_end.x);
	__m512 sec_begin_y		= set_zfvalue(-sec_begin.y);
	__m512 sec_begin_x		= set_zfvalue(sec_begin.x);
	__m512 curr_dist_square = set_zfvalue(dist * dist);
	__m512 flt_eps			= set_zfvalue(-FLT_EPSILON);
	__m512 to_unit_x; __m512 to_unit_y;
	int i;
	int res = 0;
	for (i = 0; i < mainsz; i += 16) {
		to_unit_x = load_zfvalue_unaligned(&x_coords[i]);
		to_unit_y = load_zfvalue_unaligned(&y_coords[i]);
		__m512 temp1 = mult_and_sum_zfvalues(sec_end_y, to_unit_x, mult_zfvalues(sec_end_x, to_unit_y));
		__m512 temp2 = mult_and_sum_zfvalues(sec_begin_y, to_unit_x, mult_zfvalues(sec_begin_x, to_unit_y));
		uint16_t lt_end_sec   = cmpr_less_then_zfvalues(temp1, flt_eps);
		uint16_t lt_begin_sec = cmpr_less_then_zfvalues(temp2, flt_eps);
		temp1 = mult_and_sum_zfvalues(to_unit_x, to_unit_x, mult_zfvalues(to_unit_y, to_unit_y));
		uint16_t cmp_dist = cmpr_less_then_zfvalues(subtract_zfvalues(temp1, curr_dist_square), flt_eps);
		res += number_of_bits(~lt_begin_sec & lt_end_sec & cmp_dist);
	}
	for (i = mainsz; i < n; ++i) {
		res += ((-sec_end.y * x_coords[i] + sec_end.x * y_coords[i] < -FLT_EPSILON) &&
			   !(-sec_begin.y * x_coords[i] + sec_begin.x * y_coords[i] < -FLT_EPSILON) &&
				(x_coords[i] * x_coords[i] + y_coords[i] * y_coords[i] - dist * dist < -FLT_EPSILON));
	}
	return res;
	// (-border.y * vec_to_unit.x + border.x * vec_to_unit.y < -FLT_EPSILON);
}

int calculateLine(std::vector<std::pair<int, unit>>& units_line, const vec2& sec_begin, const vec2& sec_end, const unit& curr) {
	auto it = std::lower_bound(units_line.begin(), units_line.end(), curr.position.x - curr.distance, [](std::pair<int, unit>& el, float left_border) {
		return (el.second.position.x - left_border < -FLT_EPSILON);
		});
	std::vector<float> x_coords;
	std::vector<float> y_coords;
	while (it != units_line.end() && curr.distance - std::fabs((*it).second.position.x - curr.position.x) > -FLT_EPSILON) {
		x_coords.emplace_back((*it).second.position.x - curr.position.x);
		y_coords.emplace_back((*it).second.position.y - curr.position.y);
		++it;
	}
	return isInSector(sec_end, sec_begin, curr.distance, x_coords, y_coords);
}

int calculateCurrentUnit(std::vector<std::vector<std::pair<int, unit>>>& sorted_units, const unit& curr, int idx) {
	const float k = acosf(-1.0f) / 360.0f;
	constexpr float offset = 1e6f;
	float half_fov_radian = curr.fov_deg * k; half_fov_radian = std::truncf(half_fov_radian * offset) / offset;
	float sn = sinf(half_fov_radian); sn = std::truncf(sn * offset) / offset;
	float cs = cosf(half_fov_radian); cs = std::truncf(cs * offset) / offset;
	vec2 sec_begin, sec_end;
	sec_end.x = curr.direction.x * cs - curr.direction.y * sn; sec_end.x = std::truncf(sec_end.x * offset) / offset;
	sec_end.y = curr.direction.x * sn + curr.direction.y * cs; sec_end.y = std::truncf(sec_end.y * offset) / offset;
	sn *= -1;
	sec_begin.x = curr.direction.x * cs - curr.direction.y * sn; sec_begin.x = std::truncf(sec_begin.x * offset) / offset;
	sec_begin.y = curr.direction.x * sn + curr.direction.y * cs; sec_begin.y = std::truncf(sec_begin.y * offset) / offset;
	int res = 0;
	int i = idx;
	while (i < sorted_units.size() && curr.position.y + curr.distance - sorted_units[i][0].second.position.y > FLT_EPSILON) {
		res += calculateLine(sorted_units[i], sec_begin, sec_end, curr);
		++i;
	}
	i = idx - 1;
	while (i >= 0 && curr.position.y + curr.distance - sorted_units[i][0].second.position.y > FLT_EPSILON) {
		res += calculateLine(sorted_units[i], sec_begin, sec_end, curr);
		--i;
	}
	return res;
}

void checkVisibleHelper(std::vector<std::vector<std::pair<int, unit>>>& sorted_units, std::vector<int>& result, int start, int thrds) {
	int n = sorted_units.size();

	for (int i = start; i < n; i += thrds) {
		for (int j = 0 ; j < sorted_units[i].size(); ++j) {
			result[sorted_units[i][j].first] = calculateCurrentUnit(sorted_units, sorted_units[i][j].second, i);
		}
	}
}
	
void checkVisible(const std::vector<unit>& input_units, std::vector<int>& result)
{
	SimpleTimer t;
	int n = input_units.size();
	result.resize(n, 0);
	std::vector<std::vector<std::pair<int, unit>>> sorted_units;
	{
		std::vector<std::pair<int, unit>> temp_sorted_units_with_idx(n);

		for (int i = 0; i < n; ++i) {
			temp_sorted_units_with_idx[i] = std::make_pair(i, input_units[i]);
		}

		std::sort(temp_sorted_units_with_idx.begin(), temp_sorted_units_with_idx.end(), [](std::pair<int, unit>& a, std::pair<int, unit>& b) {
			if (a.second.position.y - b.second.position.y < -FLT_EPSILON) {
				return true;
			}
			else if (a.second.position.y - b.second.position.y > FLT_EPSILON){
				return false;
			}
			else {
				return (a.second.position.x - b.second.position.x < -FLT_EPSILON);
			}
			});

		for (int i = 0; i < n; ++i) {
			std::vector<std::pair<int, unit>> temp_same_y_coord_vec(1, temp_sorted_units_with_idx[i]);

			while (i < n - 1 && temp_sorted_units_with_idx[i + 1].second.position.y - temp_sorted_units_with_idx[i].second.position.y < FLT_EPSILON &&
								temp_sorted_units_with_idx[i + 1].second.position.y - temp_sorted_units_with_idx[i].second.position.y > -FLT_EPSILON) {
				temp_same_y_coord_vec.emplace_back(temp_sorted_units_with_idx[i + 1]);
				++i;
			}
			sorted_units.emplace_back(temp_same_y_coord_vec);
		}
	}
	int thrds = std::thread::hardware_concurrency();
	thrds = std::max(thrds, 3);
	std::vector<std::thread> thrdsvec;
	for (int i = 0; i < thrds; ++i) {
		thrdsvec.emplace_back(std::thread(checkVisibleHelper, std::ref(sorted_units), std::ref(result), i, thrds + 1));
	}
	checkVisibleHelper(sorted_units, result, thrds, thrds + 1); // using main thread
	for (int i = 0; i < thrds; ++i) {
		thrdsvec[i].join();
	}
}
