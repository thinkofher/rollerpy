from abc import ABC, abstractclassmethod
import numpy as np


def numpize(func):
    '''
    Changing function ouput from list (or other iter)
    to numpy array
    '''
    def wrapper(point):
        return np.array(func(point))
    return wrapper


class Curve(ABC):

    '''
    A simple curve abstract class.
    '''

    @abstractclassmethod
    def _calcParameters(self):
        raise NotImplementedError

    def gimmePoint(self, t):
        return np.array([
            self.x[t], self.y[t], self.z[t]
            ])

    @numpize
    def returnFirstPoint(self):
        return [
            self.x[0], self.y[0], self.z[0]
        ]

    @numpize
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

    def setStartingPoint(self, sPoint):
        self.x = self.x + sPoint[0]
        self.y = self.y + sPoint[1]
        self.z = self.z + sPoint[2]


class ParametricCurve(ABC):

    '''
    Class with methods for calculating derivatives of curve.
    '''

    @abstractclassmethod
    def _calcDerivative(self):
        raise NotImplementedError

    @numpize
    def gimmeDerivative(self, t):
        return [
            self.dx[t], self.dy[t], self.dz[t]
        ]

    @numpize
    def returnFirstDerivative(self):
        return [
            self.dx[0], self.dy[0], self.dz[0]
        ]

    @numpize
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


class NoramlizedCurve(object):

    '''
    Class with methods for returing unit vectors.
    '''

    def gimmeDerivativeUnitVector(self, t):

        vector = np.array(self.gimmeDerivative(t))
        return vector/np.linalg.norm(vector)

    def returnFirstDerivativeUnitVector(self):

        vector = np.array(self.returnFirstDerivative())
        return vector/np.linalg.norm(vector)

    def returnLastDerivativeUnitVector(self):

        vector = np.array(self.returnLastDerivative())
        return vector/np.linalg.norm(vector)
