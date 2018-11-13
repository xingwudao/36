#include "dimsumv2.hpp"
#include <iostream>

using namespace std;

int main(int argc, char** argv) {
    PairSimilarityCaculator caculator(argv[1], 0.1);
    caculator.Caculate(argv[2], false);
    return 0;
}
