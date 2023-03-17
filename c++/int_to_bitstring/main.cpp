#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <algorithm>
#include <random>
#include <bitset>

using namespace std;

const __uint128_t ONE = 1;

__uint128_t string_to_int(unsigned int n, string nstring) {

	__uint128_t state, op;
	char c = '1';

	op = ONE << (n - 1);
	state = 0;

	for (auto &elem: nstring) {
			if (elem == c) {state += op;}
			op = op >> 1;
			cout << elem << endl;
		}

	return state;
}



int main() {

	string test_string = "00000000000000000000000000000000000000000000000000000000000000000000000000010010000000000000000000000000000000000000000000000000";

	__uint128_t test_int = string_to_int(128, test_string);

	cout << bitset<128>(test_int) << endl;

	return 0;
}