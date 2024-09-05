/*
* Nelson.cpp
*
* User-defined function for the Nelson dataset that is provided by NIST.
*
* https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Nelson.dat
*/
#include <math.h>  // exp
#include <string.h>  // strcpy
#include "Nelson.h"

void GetFunctionName(char* name) {
  // The function name must
  //   - begin with the letter "f",
  //   - followed by a positive integer that uniquely identifies this function,
  //   - followed by a colon.
  // The remainder of the string is optional (to describe the function).
  // The total length of the function name must be < 256 characters.
  strcpy(name, "f2: Nelson log(f2)=a1-a2*x1*exp(-a3*x2)");
}

void GetFunctionValue(double* x, double* a, double* y) {
  // Receives the x value, the fit parameters and a pointer to the y value.
  // C++ array indices are zero based
  //  - independent variables: x1=x[0], x2=x[1]
  //  - parameters: a1=a[0], a2=a[1], a3=a[2]
  *y = a[0] - a[1] * x[0] * exp(-a[2] * x[1]);
}

void GetNumParameters(int* n) {
  // There are 3 parameters: a1, a2, a3
  *n = 3;
}

void GetNumVariables(int* n) {
  // There are 2 independent variables: x1, x2
  *n = 2;
}
