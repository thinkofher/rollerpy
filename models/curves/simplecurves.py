from rollerpy.models.curve import ParametricCurve
import numpy as np


class HelixCircleParam(ParametricCurve):

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
        self.x0 = initialPosition[0]
        self.y0 = initialPosition[1]
        self.z0 = initialPosition[2]

        # Calculations
        self.t = np.linspace(self.tmin, self.tmax, n)
        self._calcParameters()
        self._calcDerivative()

    def setInitialPosition(self, initialPosition):

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
