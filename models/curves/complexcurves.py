from rollerpy.models.curves.simplecurves import (
    HelixCircleParam,
    InvHelixCircleParam
)
from rollerpy.models.curve import Curve, ParametricCurve
from rollerpy.funcs import trackTransitonCurve, curveByCurve

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.misc import derivative


_PI = np.pi

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


class TransitionHelix(NumericalDerivative, Curve, ParametricCurve):

    def __init__(
        self, percentage, lambdas, lambdae,
        helix_radius=45, helix_param=5,
        diameter=20, single_n=5, final_n=10
    ):
        # track transition parameters
        self._diamater = diameter
        self._radius = diameter/2
        self._percentage = percentage
        self._lambdas = lambdas
        self._lambdae = lambdae
        self._single_n = single_n
        self._final_n = final_n

        self._hend = 0 + (1-self._percentage/2)*_PI
        self._hstart = _PI - (1-percentage/2)*_PI

        self._helixDict = {
            'x': lambda t: self._radius*np.cos(t-0.5*_PI),
            'y': lambda t: self._radius*np.sin(t-0.5*_PI)+self._radius,
            'z': lambda t: t*0
        }
        self._dhelixDict = {
            'x': lambda t: derivative(
                self._helixDict['x'], t, dx=10**(-3), n=1),
            'y': lambda t: derivative(
                self._helixDict['y'], t, dx=10**(-3), n=1),
            'z': lambda t: derivative(
                self._helixDict['z'], t, dx=10**(-3), n=1),
        }

        # helix on curve parameters
        self._helixRadius = helix_radius
        self._helixParam = helix_param
        self._calcParameters()
        self._calcParam_t()
        self._calcDerivative()

    # TODO: create _calcParameters method
    def _calcCircleWithTran(self):

        self._tspace = np.linspace(
            self._hstart, self._hend, self._single_n
        )
        self._helixVectors = {
            'x': self._helixDict['x'](self._tspace),
            'y': self._helixDict['y'](self._tspace),
            'z': self._helixDict['z'](self._tspace)
        }

        # creating start track transition curve
        # preparing points and slopes
        _startc_sp1 = np.array([0, 0, 0])
        _startc_ss1 = np.array([1, 0, 0])*self._lambdas
        _endc_ep1 = np.array([
            self._helixDict['x'](self._hstart),
            self._helixDict['y'](self._hstart),
            self._helixDict['z'](self._hstart)
        ])
        _endc_es1 = np.array([
            self._dhelixDict['x'](self._hstart),
            self._dhelixDict['y'](self._hstart),
            self._dhelixDict['z'](self._hstart)
        ])

        # x, y, z vectors for first transition curve
        xt1, yt1, zt1 = trackTransitonCurve(
            _startc_sp1, _startc_ss1, _endc_ep1, _endc_es1,
            n=self._single_n
        )

        # creating end track transition curve
        # preparing points and slopes
        _startc_sp2 = np.array([
            self._helixDict['x'](self._hend),
            self._helixDict['y'](self._hend),
            self._helixDict['z'](self._hend)
        ])
        _start_ss2 = np.array([
            self._dhelixDict['x'](self._hend),
            self._dhelixDict['y'](self._hend),
            self._dhelixDict['z'](self._hend)
        ])
        _endc_ep2 = np.array([0, self._diamater, 0])
        _endc_es2 = np.array([-1, 0, 0])*self._lambdae

        # x, y, z vectors for second transition curve
        xt2, yt2, zt2 = trackTransitonCurve(
            _startc_sp2, _start_ss2, _endc_ep2, _endc_es2,
            n=self._single_n
        )

        # TODO: fixed this abomination
        # this is not actual very important
        # it is temporary fixing some stuff
        self._extractor = -1
        if self._percentage > 0.75:
            for key in self._helixVectors:
                self._helixVectors[key] = []
                if self._percentage < 0.9:
                    self._extractor = len(xt1)
                else:
                    self._extractor = -1

        self._ctx = np.append(
            np.append(
                xt1[:self._extractor],
                self._helixVectors['x'][:self._extractor]
            ), xt2
        )
        self._cty = np.append(
            np.append(
                yt1[:self._extractor],
                self._helixVectors['y'][:self._extractor]
            ), yt2
        )
        self._ctz = np.append(
            np.append(
                zt1[:self._extractor],
                self._helixVectors['z'][:self._extractor]
            ), zt2
        )

    def _calcParameters(self):
        self._calcCircleWithTran()

        self._tparam = np.linspace(0, 1, len(self._ctx))
        self._helixOnCurve = (
            lambda t: self._helixRadius*np.cos(
                self._helixParam*(t+0.5*_PI)
            ),
            lambda t: t,
            lambda t: self._helixRadius*np.sin(
                self._helixParam*(t+0.5*_PI)
            )
        )
        self._cubicSplines = (
            CubicSpline(self._tparam, self._ctx),
            CubicSpline(self._tparam, self._cty),
            CubicSpline(self._tparam, self._ctz)
        )

        self._finalparam = np.linspace(0, 1, self._final_n)
        self.x, self.y, self.z = curveByCurve(
            self._finalparam, self._cubicSplines, self._helixOnCurve
        )

    def _calcDerivative(self):
        self._calcParam_t()
        super()._calcDerivative()
