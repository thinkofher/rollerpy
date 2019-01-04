import numpy as np
from scipy.misc import derivative


def trackTransitonCurve(point1, slope1, point2, slope2, n=100):
        '''
        Creates u^3 paraboal from point1 to point2 with given slopes.
        '''
        t = np.linspace(0, 1, n)
        M = np.matrix(
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [1, 1, 1, 1],
                [3, 2, 1, 0]
            ],
            dtype=float
        )
        P = np.matrix(
            [
                point1,
                slope1,
                point2,
                slope2
            ],
            dtype=float
        )
        X = M**(-1) * P

        a = np.array(X[0].transpose()*(t**3))
        b = np.array(X[1].transpose()*(t**2))
        c = np.array(X[2].transpose()*(t))
        d = np.array(X[3].transpose()*(t*0+1))

        curve = a + b + c + d

        return (curve[0], curve[1], curve[2])


def printToSimulink(points):
    '''
    Printing points to copy to simulink simscape model.
    '''
    print('[')
    for x, y, z in zip(points[0], points[1], points[2]):
        print('{}, {}, {};'.format(x, y, z))
    print(']')


def frenetFromPfuncs(t, xfunc, yfunc, zfunc):
    '''
    Takes t parameter and three x, y, z parametrs funcs.
    Returns tuple with three frenet's p, n, b, unit vectors in t parameter.
    '''
    p = np.array([
        derivative(xfunc, t, dx=10e-6, n=1),
        derivative(yfunc, t, dx=10e-6, n=1),
        derivative(zfunc, t, dx=10e-6, n=1)
    ])
    p = p/np.linalg.norm(p)

    n = np.array([
        derivative(xfunc, t, dx=10e-6, n=2),
        derivative(yfunc, t, dx=10e-6, n=2),
        derivative(zfunc, t, dx=10e-6, n=2)
    ])
    n = n/np.linalg.norm(n)

    b = np.cross(p, n)
    b = b/np.linalg.norm(b)

    return (p, n, b)

# TODO: Implement curvebycurve
# TODO: create a function for easy visualization of track
