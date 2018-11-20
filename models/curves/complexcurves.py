from rollerpy.models.curves.simplecurves import (
    HelixCircleParam,
    InvHelixCircleParam,
    Line
)
from rollerpy.models.curve import Curve, ParametricCurve
from rollerpy.funcs import trackTransitonCurve

import numpy as np


class SingleLoop(Curve, ParametricCurve):

    def __init__(self, h, A, B, lw, width_param=0.5, single_e_n=50):

        # Parameters
        self._lambdaParam = B*9.5

        self._A = A
        self._B = B

        self._lw = lw
        self._h = h
        self._l1 = lw + B
        self._width_param = width_param
        self._single_e_n = single_e_n

        self._helix = HelixCircleParam(
            A=A,
            B=B,
            C=width_param,
            n=single_e_n,
            initialPosition=[self._l1, 0, h]
        )

        self._beginPoint = [0, -2*self._width_param, 0]
        self._endPoint = [
            self._l1 + lw + B,
            2*width_param+self._helix.returnLastPoint(),
            0
        ]
        self._slopeVector = [self._lambdaParam, -2*width_param, 0]

        self._calcParameters()
        self.t = np.linspace(0, 1, len(self.x))

    def _calcParameters(self):

        xt1, yt1, zt1 = trackTransitonCurve(
            self._beginPoint, self._slopeVector,
            self._helix.returnFirstPoint(), self._helix.returnFirstDerivative()
        )

        xt2, yt2, zt2 = trackTransitonCurve(
            self._helix.returnLastPoint(), self._helix.returnLastDerivative(),
            self._endPoint, self._slopeVector
        )

        self.x = np.append(
            xt1[:-1], np.append(
                self._helix.returnXarray()[:-1], xt2
            )
        )
        self.y = np.append(
            yt1[:-1], np.append(
                self._helix.returnYarray()[:-1], yt2
            )
        )
        self.z = np.append(
            zt1[:-1], np.append(
                self._helix.returnZarray()[:-1], zt2
            )
        )
