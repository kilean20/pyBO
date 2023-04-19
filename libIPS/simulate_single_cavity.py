from cavity_model import CavityModel
from elements import Cavity#, BPM, ElementContainer, Stripper
import numpy as np
from beam import Particle
from scipy.integrate import solve_ivp
import os


class simulate_single_cavity:
    def __init__(self,cav_type_name="QWR041"):
        assert cav_type_name in ["QWR041","QWR085","QWR029","QWR053","MEBTB","MGB"]
        # BPM frequency
        self.frequency = 80.5e6
        self.wavelength = 299792458 / self.frequency

        self.ttf_model = 0
        self.ttf2_model = 1
        self.realistic_model = 2
        self.model_type = self.realistic_model

        # self.stripper_q = 18
        # self.stripper_loss = 0.0e6

        self.max_step = 0.01
        self.phasing_accuracy = 1e-10

       # Cavities' types in the linac
        current_path = os.path.dirname(__file__)
       
        self.QWR041 = CavityModel(current_path + '//data//fieldmaps//Ez041.txt', 80.5e6, 0.35762, ttf_filename = current_path + '//data//ttfs//ttf041.txt')  # 5.1 MV/m
        self.QWR085 = CavityModel(current_path + '//data//fieldmaps//Ez085.txt', 80.5e6, 0.66968, ttf_filename = current_path + '//data//ttfs//ttf085.txt')  # 5.6 MV/m
        self.QWR029 = CavityModel(current_path + '//data//fieldmaps//Ez029.txt', 322e6, 0.4041, ttf_filename = current_path + '//data//ttfs//ttf029.txt')  # 7.7 MV/m
        self.QWR053 = CavityModel(current_path + '//data//fieldmaps//Ez053.txt', 322e6, 0.738081, ttf_filename = current_path + '//data//ttfs//ttf053.txt')  # 7.4 MV/m
        self.MEBTB = CavityModel(current_path + '//data//fieldmaps//EzMEBTB.txt', 80.5e6, 0.001593, ttf_filename = current_path + '//data//ttfs//ttfMEBTB.txt')  # 100 kV
        self.MGB   = CavityModel(current_path + '//data//fieldmaps//EzMGB.txt', 161e6, 0.51932792e-3, ttf_filename = current_path + '//data//ttfs//ttfMGB.txt')  # 1000 kV
        self.STRIP = CavityModel(current_path + '//data//fieldmaps//Ez000.txt', 1.0, 1.571e5, ttf_filename = current_path + '//data//ttfs//ttf000.txt', skip_points=1)  # 100 kV
        
        if cav_type_name=='QWR041':
            cav_type = self.QWR041
        elif cav_type_name=='QWR085':
            cav_type = self.QWR085
        elif cav_type_name=='MGB':
            cav_type = self.MGB
        elif cav_type_name=='QWR029':
            cav_type = self.QWR029
        elif cav_type_name=='QWR053':
            cav_type = self.QWR053
        else:
            raise ValueError("cav_type_name ",cav_type," is not recognized")
        
        self.cavity = Cavity(cav_type_name, cav_type, zc=0., offset=0., scale=1., phase=0.)
       

    def __call__(self, particle, cavity, method="realistic"):
        # integrate motion inside a cavity
        distance = cavity.length
        
        if method == "realistic":
            yinit = [particle.tau, particle.W]
            
            sol = solve_ivp(lambda z, y: self.f(z, y, particle, cavity), [-0.5 * distance, 0.5 * distance], yinit, max_step=self.max_step,
                            method='RK45')#, t_eval=np.linspace(-0.5*distance, 0.5*distance, distance/100.0))
                    
            particle.tau = sol.y[0][-1]
            particle.W = sol.y[1][-1]
            
        elif method== "TTF1":
            particle.tau = particle.tau + 2.0 * np.pi * distance / (particle.beta * self.wavelength)
            beta = self.w_to_beta(particle.W, particle.rest_energy)
            particle.W = particle.W + particle.Q * cavity.scale * cavity.field_amplitude * cavity.accelerating_voltage(
                beta) * np.cos(cavity.frequency / self.frequency * particle.tau + cavity.phase + cavity.offset)
            
        elif method == "TTF2":
            # track half cavity length
            particle.tau = particle.tau + np.pi * distance / (particle.beta * self.wavelength)
            beta = self.w_to_beta(particle.W, particle.rest_energy)
            DeltaW0 = cavity.accelerating_voltage(beta)/cavity.T(beta)
            phi = cavity.frequency / self.frequency * particle.tau + cavity.phase + cavity.offset

            # calculation of the phase shift according to J.Delayen
            if 'RFC' in cavity.name:
                tau0 = -0.5 * particle.Q * DeltaW0 / particle.W * np.sin(phi) * beta * cavity.T_prime(beta)
                particle.tau = particle.tau + tau0

            #phi = cavity.frequency / self.frequency * particle.tau + cavity.phase + cavity.offset

            particle.W = particle.W + particle.Q * cavity.scale * cavity.field_amplitude * DeltaW0 * cavity.T(beta) * np.cos(phi) + (particle.Q * cavity.scale * cavity.field_amplitude * DeltaW0)**2 / particle.W * (cavity.T2(beta) + cavity.T2s(beta)*np.sin(2.0*phi))

            # track another half cavity length
            particle.tau = particle.tau + np.pi * distance / (particle.beta * self.wavelength)
        else:
            raise ValueError("method should be in 'TTF1','TTF2', or 'realistic'")
        particle.z = particle.z + distance


    def f(self, z, y, particle, cavity):
        tau = y[0]
        energy = y[1]
        E = cavity.field
        phase = cavity.phase
        offset = cavity.offset
        scale = cavity.scale
        amplitude = cavity.field_amplitude
        harm = cavity.frequency / self.frequency
        
        gamma = 1.0 + particle.W / particle.rest_energy
        if gamma <= 1.0:
            raise ValueError("gamma is negative..")

        dtdz = 2.0 * np.pi / (self.w_to_beta(energy, particle.rest_energy) * self.wavelength)
        dwdz = particle.Q * scale * amplitude * E(z) * np.cos(harm * tau + phase + offset)
        dydz = [dtdz, dwdz]
        return dydz


    @staticmethod
    def w_to_beta(energy, rest_energy):
        gamma = 1.0 + energy / rest_energy
        beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
        return beta

    @staticmethod
    def beta_to_w(beta, rest_energy):
        gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
        energy = (gamma-1.0)*rest_energy
        return energy