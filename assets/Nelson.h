// Nelson.h

#if defined(_MSC_VER)  // Microsoft
    #define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)  // GCC
    #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
    EXPORT void GetFunctionName(char* name);
    EXPORT void GetFunctionValue(double* x, double* a, double* y);
    EXPORT void GetNumParameters(int* n);
    EXPORT void GetNumVariables(int* n);
}