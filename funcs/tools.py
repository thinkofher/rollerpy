import numpy as np


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
