import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class CavityModel:
    def __init__(self, filename, frequency, field_scale = 1, skip_points=50, ttf_filename=''):
        self.fieldmap_filename = filename
        fieldmap = np.loadtxt(self.fieldmap_filename, comments='#')
        self._z = fieldmap[::skip_points, 0] * 0.001  # convert to [m]
        self._Ez = fieldmap[::skip_points, 1]
        self.length = abs(self._z[-1] - self._z[0])
        self.field_scale = field_scale
        self.field = InterpolatedUnivariateSpline(self._z, self._Ez * self.field_scale, k=3, ext=0)
        self.frequency = frequency

        if ttf_filename:
            ttf = np.loadtxt(ttf_filename, comments='#')
            self.ttf_beta = ttf[:,0]
            self.ttf_T = ttf[:,1]
            self.ttf_T2 = ttf[:,2]
            self.ttf_T2s = ttf[:,3]
            self.ttf_TV = ttf[:, 4]
            self.T = InterpolatedUnivariateSpline(self.ttf_beta, self.ttf_T, k=3, ext=0)
            self.T2 = InterpolatedUnivariateSpline(self.ttf_beta, self.ttf_T2, k=3, ext=0)
            self.T2s = InterpolatedUnivariateSpline(self.ttf_beta, self.ttf_T2s, k=3, ext=0)
            self.T_prime = self.T.derivative()
            self.accelerating_voltage = InterpolatedUnivariateSpline(self.ttf_beta, self.ttf_TV, k=1, ext=0)
        else:
            print('Error: No TTF files assigned')

    def __str__(self):
        return 'Cavity Model from "' + self.fieldmap_filename + \
               '" with length = ' + str(round(self.length, 4)) + ' m,' \
                                                                 ' f = ' + str(self.frequency / 1e6) + ' MHz'


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cav = CavityModel('data//fieldmaps//Ez041.txt', 80.5e6)
    print(cav)
    cav = CavityModel('data//fieldmaps//Ez085.txt', 80.5e6)
    print(cav)
    cav = CavityModel('data//fieldmaps//Ez029.txt', 322e6)
    print(cav)
    cav = CavityModel('data//fieldmaps//Ez053.txt', 322e6)
    print(cav)
    cav = CavityModel('data//fieldmaps//EzMEBTB.txt', 80.5e6,skip_points=50)
    print(cav)
    cav = CavityModel('data//fieldmaps//EzMGB.txt', 161e6,skip_points=50)
    print(cav)

    z = np.arange(-0.7,0.7,0.01)
    plt.plot(cav._z, cav._Ez, '.-')
    plt.plot(z,cav.field(z),'x-')
    plt.xlabel('z (m)')
    plt.ylabel('Ez (V/m)')
    plt.show()

