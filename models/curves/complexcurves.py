from rollerpy.models.curves.simplecurves import (
    HelixCircleParam,
    InvHelixCircleParam,
    Line
)
from rollerpy.models.curve import Curve, ParametricCurve
from rollerpy.funcs import trackTransitonCurve

import numpy as np
from scipy.interpolate import CubicSpline


class SingleLoop(Curve, ParametricCurve):

    def __init__(
        self, h, A, B, lw, width_param=0.5, single_e_n=50,
        slopeCoeffStart=9.5, slopeCoeffEnter=2
    ):

        # Parameters
        self._lambdaParam = B*slopeCoeffStart

        self._A = A
        self._B = B

        self._lw = lw
        self._h = h
        self._l1 = lw + B
        self._width_param = width_param
        self._single_e_n = single_e_n

        self._slopeCoeff = slopeCoeffEnter

        self._helix = HelixCircleParam(
            A=A,
            B=B,
            C=width_param,
            n=single_e_n,
            initialPosition=[self._l1, 0, h],
            tmin=(1/5)*np.pi,
            tmax=(4/5)*np.pi
        )

        self._beginPoint = [0, -2*self._width_param, 0]
        self._endPoint = [
            self._l1 + lw + B,
            2*width_param+self._helix.returnLastPoint()[1],
            0
        ]
        self._slopeVector = [self._lambdaParam, -2*width_param, 0]

        self._calcParameters()
        self._calcParam_t()
        self._calcDerivative()

    def _calcParam_t(self):
        self.t = np.linspace(0, 1, len(self.x))

    def _calcParameters(self):

        xt1, yt1, zt1 = trackTransitonCurve(
            self._beginPoint, self._slopeVector,
            self._helix.returnFirstPoint(),
            np.array(self._helix.returnFirstDerivative())*self._slopeCoeff
        )

        xt2, yt2, zt2 = trackTransitonCurve(
            self._helix.returnLastPoint(),
            np.array(self._helix.returnLastDerivative())*self._slopeCoeff,
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

    def _calcDerivative(self):
        self._xCubicFunc = CubicSpline(self.t, self.x)
        self._yCubicFunc = CubicSpline(self.t, self.y)
        self._zCubicFunc = CubicSpline(self.t, self.z)

        self.dx = self._xCubicFunc(self.t, 1)
        self.dy = self._yCubicFunc(self.t, 1)
        self.dz = self._zCubicFunc(self.t, 1)


class DoubleLoop(SingleLoop):

    def _calcParameters(self):
        super()._calcParameters()

        self._invhelix = InvHelixCircleParam(
            A=self._A,
            B=self._B,
            C=self._width_param,
            n=self._single_e_n,
            initialPosition=[self._l1*3, 0, self._h],
            tmin=(1/5)*np.pi,
            tmax=(4/5)*np.pi
        )

        self._endOfSecondLoop = [
            (self._l1+self._lw+self._B)*2, self.returnFirstPoint()[1], 0
        ]
        self._slopeVectorEnd = [
            self._lambdaParam, 2*self._width_param, 0
        ]

        xt1, yt1, zt1 = trackTransitonCurve(
            self.returnLastPoint(),
            self._slopeVector,
            self._invhelix.returnFirstPoint(),
            np.array(self._invhelix.returnFirstDerivative())*self._slopeCoeff
        )

        xt2, yt2, zt2 = trackTransitonCurve(
            self._invhelix.returnLastPoint(),
            np.array(self._invhelix.returnLastDerivative())*self._slopeCoeff,
            self._endOfSecondLoop,
            self._slopeVectorEnd
        )

        self.x = np.append(
            self.x[:-1], np.append(
                xt1[:-1], np.append(
                    self._invhelix.returnXarray()[:-1], xt2
                )
            )
        )

        self.y = np.append(
            self.y[:-1], np.append(
                yt1[:-1], np.append(
                    self._invhelix.returnYarray()[:-1], yt2
                )
            )
        )

        self.z = np.append(
            self.z[:-1], np.append(
                zt1[:-1], np.append(
                    self._invhelix.returnZarray()[:-1], zt2
                )
            )
        )
