/*
* Roszman1.cpp
*
* User-defined function for the Roszman1 dataset that is provided by NIST.
*
* https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Roszman1.dat
*/
#include <math.h>  // atan
#include <string.h>  // strcpy_s
#include "Roszman1.h"

void GetFunctionName(char* name) {
  // The function name must
  //   - begin with the letter "f",
  //   - followed by a positive integer that uniquely identifies this function,
  //   - followed by a colon.
  // The remainder of the string is optional (to describe the function).
  strcpy_s(name, 255, "f1: Roszman1 f1=a1-a2*x-arctan(a3/(x-a4))/pi");
}

void GetFunctionValue(double* x, double* a, double* y) {
  // Receives the x value, the fit parameters and a pointer to the y value.
  // C++ array indices are zero based (i.e., a1=a[0] and x=x[0])
  *y = a[0] - a[1] * x[0] - atan(a[2] / (x[0] - a[3])) / pi;
}

void GetNumParameters(int* n) {
  // There are 4 parameters: a1, a2, a3, a4
  *n = 4;
}

void GetNumVariables(int* n) {
  // There is only 1 independent variable: x
  *n = 1;
}
