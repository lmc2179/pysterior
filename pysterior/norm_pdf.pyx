cdef float pi = 3.141592653589793

cdef extern from "math.h":
    float log(float x)

def lognormpdf(float x, float mean, float sd):
    cdef float var = sd**2
    cdef float denom = (2*pi*var)**.5
    cdef float num = -(float(x)-float(mean))**2/(2*var)
    return num - log(denom)
