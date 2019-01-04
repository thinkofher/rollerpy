import numpy as np
from scipy.misc import derivative


_GLOBALSYSTEM = (
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1])
)


def trackTransitonCurve(point1, slope1, point2, slope2, n=100):
        '''
        Creates u^3 paraboal [x, y, z] vector from
        point1 to point2 with given slopes.
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


def _calcCos(veca, vecb):
    '''
    Calculating cos between veca and vecb.
    '''
    return np.dot(veca, vecb)/(
        np.linalg.norm(veca)*np.linalg.norm(vecb)
    )


def curveByCurve(linspace, basefuncs, curfuncs):
    '''
    Returns [x, y, z] vector of curfuncs curve in local
    coordinate system of basefuncs in given linspace.
    '''
    xnew = np.array([])
    ynew = np.array([])
    znew = np.array([])
    xb, yb, zb = _GLOBALSYSTEM
    for t in linspace:
        posbase = np.array([
            basefuncs[0](t), basefuncs[1](t), basefuncs[2](t)
        ])
        poscurv = np.array([
            curfuncs[0](t), curfuncs[1](t), curfuncs[2](t)
        ])
        p, n, b = frenetFromPfuncs(t, basefuncs[0], basefuncs[1], basefuncs[2])

        transmatrix = np.array(
            [
                [_calcCos(p, xb), _calcCos(n, xb), _calcCos(b, xb)],
                [_calcCos(p, yb), _calcCos(n, yb), _calcCos(b, yb)],
                [_calcCos(p, zb), _calcCos(n, zb), _calcCos(b, zb)],
            ]
        )
        transmatrix = np.array(
            [
                [_calcCos(p, xb), _calcCos(p, yb), _calcCos(p, zb)],
                [_calcCos(n, xb), _calcCos(n, yb), _calcCos(n, zb)],
                [_calcCos(b, xb), _calcCos(b, yb), _calcCos(b, zb)],
            ]
        )

        rotated = np.matmul(transmatrix, poscurv)

        xnew = np.append(xnew, rotated[0] + posbase[0])
        ynew = np.append(ynew, rotated[1] + posbase[1])
        znew = np.append(znew, rotated[2] + posbase[2])

    return (xnew, ynew, znew)

# TODO: create a function for easy visualization of track
