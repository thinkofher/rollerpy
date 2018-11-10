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
            self.x[1], self.y[1], self.z[1]
        ]

    def returnLastPoint(self):
        return [
            self.x[-1], self.y[-1], self.z[-1]
        ]

    @abstractclassmethod
    def returnXarray(self):
        raise NotImplementedError

    @abstractclassmethod
    def returnYarray(self):
        raise NotImplementedError

    @abstractclassmethod
    def returnZarray(self):
        raise NotImplementedError


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
            self.dx[1], self.dy[1], self.dz[1]
        ]

    def returnLastDerivative(self):
        return [
            self.dx[-1], self.dy[-1], self.dz[-1]
        ]
