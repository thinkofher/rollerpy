from abc import ABC, abstractclassmethod


class Curve(ABC):

    @abstractclassmethod
    def _calcParameters(self):
        raise NotImplementedError

    def gimmePoint(self, t):
        return [
            self.x[t], self.y[t], self.z[t]
            ]

    def returnFirstPoint(self):
        return [
            self.x[0], self.y[0], self.z[0]
        ]

    def returnLastPoint(self):
        return [
            self.x[-1], self.y[-1], self.z[-1]
        ]

    def returnXarray(self):
        return self.x

    def returnYarray(self):
        return self.y

    def returnZarray(self):
        return self.z

    def returnTparam(self):
        return self.t


class ParametricCurve(Curve, ABC):

    @abstractclassmethod
    def _calcDerivative(self):
        raise NotImplementedError

    def gimmeDerivative(self, t):
        return [
            self.dx[t], self.dy[t], self.dz[t]
        ]

    def returnFirstDerivative(self):
        return [
            self.dx[0], self.dy[0], self.dz[0]
        ]

    def returnLastDerivative(self):
        return [
            self.dx[-1], self.dy[-1], self.dz[-1]
        ]

    def returndXarray(self):
        return self.dx

    def returndYarray(self):
        return self.dy

    def returndZarray(self):
        return self.dz
