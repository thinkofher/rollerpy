from rollerpy.models.curve import Curve, ParametricCurve, NoramlizedCurve
import numpy as np


class HelixCircleParam(Curve, ParametricCurve, NoramlizedCurve):

    def __init__(
        self, A, B, C=1, tmin=0, tmax=np.pi, n=100, initialPosition=[0, 0, 0]
    ):
        # Helix parameters
        self.A = A
        self.B = B
        self.C = C
        self.tmin = tmin
        self.tmax = tmax

        # Initial positions
        self._setInitialPosition(initialPosition)

        # Calculations
        self.t = np.linspace(self.tmin, self.tmax, n)
        self._calcParameters()
        self._calcDerivative()

    def _setInitialPosition(self, initialPosition):

        self.x0 = initialPosition[0]
        self.y0 = initialPosition[1]
        self.z0 = initialPosition[2]

    def _calcParameters(self):
        self.x = self.B*np.cos(self.t) + self.x0
        self.y = self.C*self.t + self.y0 - self.tmin
        self.z = self.A*np.sin(self.t) + self.z0

    def _calcDerivative(self):
        self.dx = -1*self.B*np.sin(self.t)
        self.dy = self.t*0 + self.C
        self.dz = self.A*np.cos(self.t)


class InvHelixCircleParam(HelixCircleParam):

    def _calcParameters(self):
        self.x = self.B*np.cos(self.t) + self.x0
        self.y = -self.C*(self.t - self.tmax) + self.y0 - self.tmin
        self.z = self.A*np.sint(self.t) + self.z0

    def _calcDerivative(self):
        self.dx = -1*self.B*np.sin(self.t)
        self.dy = self.t*0 - self.C
        self.dz = self.A*np.cos(self.t)


class Line(ParametricCurve, NoramlizedCurve):

    def __init__(self, point1, point2, tmin=0, tmax=1, n=100):

        self._point1 = point1
        self._point2 = point2

        self._v = [
            (point2[0] - point1[0])/tmax,
            (point2[1] - point1[1])/tmax,
            (point2[2] - point1[2])/tmax
        ]

        self.t = np.linspace(tmin, tmax, n)

        self._calcParameters()
        self._calcDerivative()

    def _calcParameters(self):
        self.x = self._point1[0] + self.t*self._v[0]
        self.y = self._point1[1] + self.t*self._v[1]
        self.z = self._point1[2] + self.t*self._v[2]

    def _calcDerivative(self):
        self.dx = 0*self.t + (self._point2[0] - self._point1[0])/self.t[-1]
        self.dy = 0*self.t + (self._point2[1] - self._point1[1])/self.t[-1]
        self.dz = 0*self.t + (self._point2[2] - self._point1[2])/self.t[-1]
