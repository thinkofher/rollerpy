from rollerpy.models.curves.simplecurves import (
    HelixCircleParam,
    InvHelixCircleParam,
    Line
)
from rollerpy.models.curve import Curve, ParametricCurve
from rollerpy.funcs import trackTransitonCurve

import numpy as np
from scipy.interpolate import CubicSpline


class NumericalDerivative(object):

    def _calcParam_t(self):
        try:
            self.t = np.linspace(0, self._tparam_max, len(self.x))
        except AttributeError:
            self.t = np.linspace(0, 1, len(self.x))

    def _calcDerivative(self):
        self._xCubicFunc = CubicSpline(self.t, self.x)
        self._yCubicFunc = CubicSpline(self.t, self.y)
        self._zCubicFunc = CubicSpline(self.t, self.z)

        self.dx = self._xCubicFunc(self.t, 1)
        self.dy = self._yCubicFunc(self.t, 1)
        self.dz = self._zCubicFunc(self.t, 1)


class SingleLoop(NumericalDerivative, Curve, ParametricCurve):

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

    def _calcParameters(self):

        xt1, yt1, zt1 = trackTransitonCurve(
            self._beginPoint, self._slopeVector,
            self._helix.returnFirstPoint(),
            np.array(self._helix.returnFirstDerivative())*self._slopeCoeff,
            n=self._single_e_n
        )

        xt2, yt2, zt2 = trackTransitonCurve(
            self._helix.returnLastPoint(),
            np.array(self._helix.returnLastDerivative())*self._slopeCoeff,
            self._endPoint, self._slopeVector,
            n=self._single_e_n
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
        super()._calcDerivative()


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
            np.array(self._invhelix.returnFirstDerivative())*self._slopeCoeff,
            n=self._single_e_n
        )

        xt2, yt2, zt2 = trackTransitonCurve(
            self._invhelix.returnLastPoint(),
            np.array(self._invhelix.returnLastDerivative())*self._slopeCoeff,
            self._endOfSecondLoop,
            self._slopeVectorEnd,
            n=self._single_e_n
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


class Hill(NumericalDerivative, Curve, ParametricCurve):

    def __init__(
        self,
        l_e1, l_e2, h,
        lamb_s, lamb_m, lamb_e,
        single_e_n=15, h_start=0, h_end=0
    ):

        # Points
        self._startingPoint = [
            h_start, 0, 0
        ]
        self._middlePoint = [
            l_e1, 0, h
        ]
        self._endingPoint = [
            l_e1+l_e2, 0, h_end
        ]

        # Slopes
        self._startingSlope = [
            lamb_s, 0, 0
        ]
        self._middleSlope = [
            lamb_m, 0, 0
        ]
        self._endSlope = [
            lamb_e, 0, 0
        ]

        self._single_e_n = single_e_n
        self._calcParameters()
        self._calcParam_t()
        self._calcDerivative()

    def _calcParameters(self):

        x1, y1, z1 = trackTransitonCurve(
            self._startingPoint,
            self._startingSlope,
            self._middlePoint,
            self._middleSlope,
            n=self._single_e_n
        )
        x2, y2, z2 = trackTransitonCurve(
            self._middlePoint,
            self._middleSlope,
            self._endingPoint,
            self._endSlope
        )

        # every parameter without last element of vector
        # to make possible to interpolate vectors in future
        self.x = np.append(x1[:-1], x2)
        self.y = np.append(y1[:-1], y2)
        self.z = np.append(z1[:-1], z2)

    def _calcDerivative(self):
        super()._calcDerivative()
