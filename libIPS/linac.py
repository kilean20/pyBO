import datetime

from cavity_model import CavityModel
from elements import Cavity, BPM, ElementContainer, Stripper
import numpy as np
from beam import Particle
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import multiprocessing
import os


class Linac:
    def __init__(self, lattice_file):
        # BPM frequency
        self.frequency = 80.5e6
        self.wavelength = 299792458 / self.frequency

        self.ttf_model = 0
        self.ttf2_model = 1
        self.realistic_model = 2
        self.model_type = self.realistic_model

        self.stripper_q = 18
        self.stripper_loss = 0.0e6

        self.max_step = 0.01
        self.phasing_accuracy = 1e-10

        self.elements = ElementContainer()
        self.cavities = ElementContainer()
        self.bpms = ElementContainer()

        current_path = os.path.dirname(__file__)

        # Cavities' types in the linac
        self.QWR041 = CavityModel(current_path + '//data//fieldmaps//Ez041.txt', 80.5e6, 0.35762, ttf_filename = current_path + '//data//ttfs//ttf041.txt')  # 5.1 MV/m
        self.QWR085 = CavityModel(current_path + '//data//fieldmaps//Ez085.txt', 80.5e6, 0.66968, ttf_filename = current_path + '//data//ttfs//ttf085.txt')  # 5.6 MV/m
        self.QWR029 = CavityModel(current_path + '//data//fieldmaps//Ez029.txt', 322e6, 0.4041, ttf_filename = current_path + '//data//ttfs//ttf029.txt')  # 7.7 MV/m
        self.QWR053 = CavityModel(current_path + '//data//fieldmaps//Ez053.txt', 322e6, 0.738081, ttf_filename = current_path + '//data//ttfs//ttf053.txt')  # 7.4 MV/m
        self.MEBTB = CavityModel(current_path + '//data//fieldmaps//EzMEBTB.txt', 80.5e6, 0.001593, ttf_filename = current_path + '//data//ttfs//ttfMEBTB.txt')  # 100 kV
        self.MGB   = CavityModel(current_path + '//data//fieldmaps//EzMGB.txt', 161e6, 0.51932792e-3, ttf_filename = current_path + '//data//ttfs//ttfMGB.txt')  # 1000 kV
        self.STRIP = CavityModel(current_path + '//data//fieldmaps//Ez000.txt', 1.0, 1.571e5, ttf_filename = current_path + '//data//ttfs//ttf000.txt', skip_points=1)  # 100 kV

        self.load_lattice_from_file(lattice_file)

    def load_lattice_from_file(self, filename):
        self.elements = ElementContainer()
        self.cavities = ElementContainer()
        self.bpms = ElementContainer()
        self.strippers = ElementContainer()

        zc = np.loadtxt(filename, comments=['N','#'], usecols=[1]) * 0.001  # convert to [m]
        name = np.loadtxt(filename, comments=['N','#'], dtype=np.str, usecols=[2])
        offset = np.loadtxt(filename, comments=['N','#'], usecols=[3]) * np.pi / 180.0
        scale = np.loadtxt(filename, comments=['N','#'], usecols=[4])
        lolo = np.loadtxt(filename, comments=['N','#'], usecols=[5])
        lo = np.loadtxt(filename, comments=['N','#'], usecols=[6])
        hi = np.loadtxt(filename, comments=['N','#'], usecols=[7])
        hihi = np.loadtxt(filename, comments=['N','#'], usecols=[8])

        for i in range(len(zc)):
            if 'STR' in name[i]:
                element = Stripper(name[i], zc[i], offset[i], 0.0)
                self.strippers.append(element)
                #self.cavities.append(element)

            elif 'BPM' in name[i]:
                element = BPM(name[i], zc[i], offset[i], 0.0)
                self.bpms.append(element)

            else:

                # detect cavity type
                if 'MEBT:RFC' in name[i]:
                    cav_type = self.MEBTB
                elif 'LS1_CA' in name[i]:
                    cav_type = self.QWR041
                elif 'LS1_CB' in name[i]:
                    cav_type = self.QWR085
                elif 'FS1_CH' in name[i]:
                    cav_type = self.QWR085
                elif 'FS1_MGB' in name[i]:
                    cav_type = self.MGB
                elif 'LS2_CC' in name[i]:
                    cav_type = self.QWR029
                elif 'LS2_CD' in name[i]:
                    cav_type = self.QWR053
                elif 'FS2_CG' in name[i]:
                    cav_type = self.QWR053
                elif 'LS3_CD' in name[i]:
                    cav_type = self.QWR053
                else:
                    cav_type = self.STRIP
                    # error case

                element = Cavity(name[i], cav_type, zc[i], offset[i], scale[i])
                element.lolo = lolo[i]
                element.lo = lo[i]
                element.hi = hi[i]
                element.hihi = hihi[i]
                self.cavities.append(element)

            self.elements.append(element)

    def save_lattice_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('N\tz(mm)\tName\toffset(deg)\tscale\tLOLO\tLO\tHI\tHIHI\n')
            for i, element in enumerate(self.elements):
                z = element.zc * 1000.0
                offset = round(wrap(element.offset * 180.0 / np.pi, limit=180.0), 3)
                scale = round(element.scale, 6)
                f.write(str(i) + '\t' + str(z) + '\t' + element.name + '\t' + str(offset) + '\t' + str(scale) +'\t' + str(element.lolo) +'\t' + str(element.lo) + '\t' + str(element.hi) + '\t' + str(element.hihi) + '\n')

    def save_profile_to_file(self, filename, particle):
        with open(filename, 'w') as f:
            f.write('N\tName\tE*q/A(MV/m)\tSynch.phase(deg)\n')
            for i, cav in enumerate(self.cavities):
                phys = round(wrap(cav.synchronous_phase * 180.0 / np.pi, limit=180.0), 3)
                if self.strippers:
                    if cav.zc < self.strippers[0].zc:
                        field = round(cav.field_amplitude * particle.initial_Q / particle.A, 6)
                    else:
                        field = round(cav.field_amplitude * particle.stripped_Q / particle.A, 6)
                else:
                    field = round(cav.field_amplitude * particle.stripped_Q / particle.A, 6)               
                f.write(str(i) + '\t' + cav.name + '\t' + str(field) + '\t' + str(phys) + '\n')

    def load_profile_from_file(self, filename, particle):
        names = np.loadtxt(filename, comments='N', dtype=np.str, usecols=[1])
        fields = np.loadtxt(filename, comments='N', usecols=[2])
        phases = np.loadtxt(filename, comments='N', usecols=[3])
        for cav in self.cavities:
            if cav.name in names:
                index = list(names).index(cav.name)
                if self.strippers:
                    if cav.zc < self.strippers[0].zc:
                        cav.field_amplitude = fields[index] * particle.A / particle.initial_Q
                    else:
                        cav.field_amplitude = fields[index] * particle.A / particle.stripped_Q
                else:
                    cav.field_amplitude = fields[index] * particle.A / particle.initial_Q
                cav.synchronous_phase = phases[index] * np.pi / 180.0

    def save_tune_to_file(self, filename, particle):
        with open(filename, 'w') as f:
            f.writelines('##################################################################\n')
            f.writelines('# Tune file for the Instant Phase Setting application\n')
            f.writelines('# Beam A: ' + str(particle.A)+'\n')
            f.writelines('# Beam Q: ' + str(particle.initial_Q)+'\n')
            f.writelines('# Beam Q stripped: ' + str(particle.stripped_Q)+'\n')
            f.writelines('# Beam init energy (eV/u): ' + str(particle.input_energy_in_eV_u)+'\n')
            f.writelines('# Beam init phase (deg): ' + str(particle.initial_phase)+'\n')
            f.writelines('# Timestamp: ' + str(datetime.datetime.now().strftime("%Y/%m/%d, %H:%M:%S"))+'\n')
            f.writelines('##################################################################\n')
            f.write('N\tz(mm)\tName\toffset(deg)\tscale\tLOLO\tLO\tHI\tHIHI\tphase(deg)\tfield(MV/m)\tSync.phase(deg)\n')
            for i, element in enumerate(self.elements):
                z = element.zc * 1000.0
                offset = round(wrap(element.offset * 180.0 / np.pi, limit=180.0), 3)
                scale = round(element.scale, 6)
                if 'BPM' in element.name:
                    phase = round(wrap(element.phase * 180.0 / np.pi, limit=90.0), 3)
                else:
                    phase = round(wrap(element.phase * 180.0 / np.pi, limit=180.0), 3)
                if 'RFC' in element.name:
                    f.write(str(i) + '\t' + str(z) + '\t' + element.name + '\t' + str(offset) + '\t' + str(scale) +'\t' + str(element.lolo) +'\t' + str(element.lo) + '\t' + str(element.hi) + '\t' + str(element.hihi) + '\t' + str(phase) + '\t' + str(round(element.field_amplitude, 6))+ '\t' + str(round(wrap(element.synchronous_phase * 180.0 / np.pi, limit=180.0), 3))+'\n')
                else:
                    f.write(str(i) + '\t' + str(z) + '\t' + element.name + '\t' + str(offset) + '\t' + str(
                        scale) + '\t' + str(element.lolo) + '\t' + str(element.lo) + '\t' + str(
                        element.hi) + '\t' + str(element.hihi) + '\t' + str(phase) + '\t' + str(
                        0.0) + '\t' + str(0.0) + '\n')

    def load_tune_from_file(self, filename, particle):
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Beam A:' in line:
                    BeamA_line = line
                    particle.A = float(BeamA_line.split(":")[1])
                elif 'Beam Q:' in line:
                    BeamQ_line = line
                    particle.initial_Q = float(BeamQ_line.split(":")[1])
                    particle.Q = float(BeamQ_line.split(":")[1])
                elif 'Beam Q stripped:' in line:
                    BeamQstripped_line = line
                    particle.stripped_Q = float(BeamQstripped_line.split(":")[1])
                elif 'Beam init energy' in line:
                    BeamW_line = line
                    particle.input_energy_in_eV_u = float(BeamW_line.split(":")[1])
                elif 'Beam init phase' in line:
                    BeamPhase_line = line
                    particle.initial_phase = float(BeamPhase_line.split(":")[1])
                else:
                    pass


        names = np.loadtxt(filename, comments=['N','#'], dtype=np.str, usecols=[2])
        fields = np.loadtxt(filename, comments=['N','#'], usecols=[10])
        phases = np.loadtxt(filename, comments=['N','#'], usecols=[9])
        sync_phases = np.loadtxt(filename, comments=['N','#'], usecols=[11])
        for cav in self.cavities:
            if cav.name in names:
                index = list(names).index(cav.name)
                cav.field_amplitude = fields[index]
                cav.synchronous_phase = sync_phases[index] * np.pi / 180.0
                cav.phase = phases[index] * np.pi / 180.0

    def print_bpm_phases(self):
        for bpm in self.bpms:
            print(bpm.name, bpm.phase)

    def print_cavities_phases(self):
        for cavity in self.cavities:
            print(cavity.name, cavity.phase, cavity.scale)

    def track_particle(self, particle, n_start=0, n_end=100000, debug_mode = False):

        # Initialization
        if n_start == 0:
            particle.z = 0.0
            particle.tau = particle.initial_phase * np.pi / 180.0
            particle.W = particle.input_energy_in_eV
            particle.Q = particle.initial_Q

            particle.zp = np.zeros(len(self.elements))
            particle.Wp = np.zeros(len(self.elements))
            particle.taup = np.zeros(len(self.elements))
            particle.Qp = np.zeros(len(self.elements))

        else:
            particle.z = particle.zp[n_start-1]
            particle.tau = particle.taup[n_start-1]
            particle.W = particle.Wp[n_start-1]
            particle.Q = particle.Qp[n_start-1]

        if debug_mode:
            with open('debug_data.txt', 'w') as debug_file:
                debug_file.write('z\ttau\tW\n')
                #debug_file.write(str(particle.z)+'\t'+str(particle.tau)+'\t'+str(particle.W)+'\n')

        # Element-by-element tracking
        for i, element in enumerate(self.elements):
            if i < n_start:
                continue

            if i > n_end:
                break

            if element in self.bpms:
                self.track_to_bpm(particle, element, debug_mode)
            elif element in self.strippers:
                self.track_to_stripper(particle, element, debug_mode)
            else:
                self.track_to_cav(particle, element, debug_mode)
                self.track_in_cav(particle, element, debug_mode)

            particle.Qp[i] = particle.Q
            particle.zp[i] = particle.z
            particle.Wp[i] = particle.W
            particle.taup[i] = particle.tau

        return particle.zp,particle.taup,particle.Wp

    def track_to_bpm(self, particle, element, debug_mode=False):
        # move to the bpm position, zc - bpm center, z - current position
        distance = element.zc - particle.z
        particle.z = particle.z + distance
        particle.tau = particle.tau + 2.0 * np.pi * distance / (particle.beta * self.wavelength)
        element.phase = particle.tau + element.offset
        if debug_mode:
            with open('debug_data.txt', 'a+') as debug_file:
                debug_file.write(str(particle.z)+'\t'+str(particle.tau)+'\t'+str(particle.W)+'\n')

    def track_to_stripper(self, particle, element, debug_mode=False):
        # move to the bpm position, zc - bpm center, z - current position
        distance = element.zc - particle.z
        particle.z = particle.z + distance
        particle.tau = particle.tau + 2.0 * np.pi * distance / (particle.beta * self.wavelength)
        element.phase = particle.tau + element.offset
        
        particle.W = particle.W - self.stripper_loss * particle.A
        #particle.Q = particle.stripped_Q
        if debug_mode:
            with open('debug_data.txt', 'a+') as debug_file:
                debug_file.write(str(particle.z)+'\t'+str(particle.tau)+'\t'+str(particle.W)+'\n')


    def track_to_cav(self, particle, cavity, debug_mode=False):
        # move to the cavity entrance, zc - cavity center, z - current position
        distance = cavity.zc - particle.z - cavity.length / 2.0
        particle.z = particle.z + distance
        particle.tau = particle.tau + 2.0 * np.pi * distance / (particle.beta * self.wavelength)
        if debug_mode:
            with open('debug_data.txt', 'a+') as debug_file:
                debug_file.write(str(particle.z)+'\t'+str(particle.tau)+'\t'+str(particle.W)+'\n')

    def track_in_cav(self, particle, cavity, debug_mode=False):
        # integrate motion inside a cavity
        distance = cavity.length
        yinit = [particle.tau, particle.W]

        #if cavity.field_amplitude > 0.0:
        #    cavity.turn_on()
        #else:
        #    cavity.turn_off()

        particle.Q = particle.initial_Q

        if self.strippers:
            if cavity.zc > self.strippers[0].zc:
                particle.Q = particle.stripped_Q
            else:
                particle.Q = particle.initial_Q


        if cavity.on_off_status == cavity.is_on:
            if self.model_type == self.realistic_model:

                sol = solve_ivp(lambda z, y: self.f(z, y, particle, cavity), [-0.5 * distance, 0.5 * distance], yinit, max_step=self.max_step,
                                method='RK45')#, t_eval=np.linspace(-0.5*distance, 0.5*distance, distance/100.0))
                if debug_mode:
                    with open('debug_data.txt', 'a+') as debug_file:
                        for i,z in enumerate(sol.t):
                            debug_file.write(str(particle.z+z+0.5*distance) + '\t' + str(sol.y[0][i]) + '\t' + str(sol.y[1][i]) + '\n')

                particle.tau = sol.y[0][-1]
                particle.W = sol.y[1][-1]

            elif self.model_type == self.ttf_model:

                particle.tau = particle.tau + 2.0 * np.pi * distance / (particle.beta * self.wavelength)
                beta = self.w_to_beta(particle.W, particle.rest_energy)
                particle.W = particle.W + particle.Q * cavity.scale * cavity.field_amplitude * cavity.accelerating_voltage(
                    beta) * np.cos(cavity.frequency / self.frequency * particle.tau + cavity.phase + cavity.offset)

            elif self.model_type == self.ttf2_model:

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

                print('Error: No beam dynamics model selected')
        else:
            particle.tau = particle.tau + 2.0 * np.pi * distance / (particle.beta * self.wavelength)

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

        dtdz = 2.0 * np.pi / (self.w_to_beta(energy, particle.rest_energy) * self.wavelength)
        dwdz = particle.Q * scale * amplitude * E(z) * np.cos(harm * tau + phase + offset)
        dydz = [dtdz, dwdz]
        return dydz

    def phase_scan(self):
        pass

    def max_acceleration_phase(self, cavity_index, particle, n_end=100000):
        result = minimize_scalar(self.deceleration, bounds=(-180, 180), method='bounded',
                                 args=(cavity_index, particle, n_end), options={'xatol': self.phasing_accuracy})
        return result.x

    def deceleration(self, phase, cavity_index, particle, n_end=100000):
        # save current phase
        phase0 = self.cavities[cavity_index].phase
        index = self.elements.index(self.cavities[cavity_index])
        self.cavities[cavity_index].phase = phase * np.pi / 180.0
        self.track_particle(particle, n_start = index, n_end=index + 10)
        # restore current phase
        self.cavities[cavity_index].phase = phase0
        return particle.Wp[index - 1] - particle.Wp[index]

    def get_phys(self, cavity_index, particle, n_end=100000):
        crest_phase = self.max_acceleration_phase(cavity_index, particle, n_end)
        #print(self.cavities[cavity_index].name,crest_phase-self.cavities[cavity_index].phase*180/np.pi)
        self.cavities[cavity_index].synchronous_phase = wrap(self.cavities[cavity_index].phase - crest_phase * np.pi / 180.0, limit = np.pi)
        return 0

    def set_phys(self, cavity_index, particle, n_end=100000):
        index = self.elements.index(self.cavities[cavity_index])
        if cavity_index+1 < len(self.cavities):
            index_next = self.elements.index(self.cavities[cavity_index+1])
        else:
            index_next = 100000
        self.cavities[cavity_index].turn_on()
        phase = self.cavities[cavity_index].synchronous_phase * 180.0 / np.pi
        #self.track_particle(particle, n_end=n_end)
        crest_phase = self.max_acceleration_phase(cavity_index, particle, n_end=index_next)
        self.cavities[cavity_index].phase = (crest_phase + phase) * np.pi / 180.0
        self.track_particle(particle, n_start=index)
        return 0

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


# END OF CLASS

def wrap_single(phase, limit=180.0):
    while phase >= limit:
        phase = phase - limit * 2.0
    while phase < -limit:
        phase = phase + limit * 2.0
    return phase


def wrap(phases, limit=180.0):
    if isinstance(phases, (int, float)):
        return wrap_single(phases, limit)
    else:
        phases = list(phases)
        phase_out = []
    for i, phase in enumerate(phases):
        phase_out.append(wrap_single(phase, limit))
    return phase_out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar
    import time

    LS = Linac('data/lattices/test_lattice.ltc')
    LS.model_type = LS.realistic_model
    LS.model_type = LS.ttf2_model
    #LS.model_type = LS.ttf_model
    ion = Particle(238, 33, 0.504e6)

    LS.load_profile_from_file('data/lattices/profile_test.vpr', ion)

    phase_linac = True
    if phase_linac:
        for cavity in LS.cavities:
            cavity.turn_off()

        LS.track_particle(ion)

        t0 = time.time()
        for cavity_index, cav in enumerate(LS.cavities):
            LS.cavities[cavity_index].turn_on()
            # LS.cavities[cavity_index].field_amplitude = 5.1
            # LS.cavities[cavity_index].synchronous_phase = -30 * np.pi/180
            LS.set_phys(cavity_index, ion)
            #LS.track_particle(ion)
            print(cav.name, ion.Wp[-1] / ion.A / 1e6)
        t1 = time.time()
        print('Phasing time:', t1 - t0)

    LS.track_particle(ion)
    plt.plot(ion.zp,ion.Wp/ion.A/1e6)
    plt.bar(LS.cavities.zc, np.ones(len(LS.cavities)),width=0.2,color='k')
    plt.show()

    LS.save_lattice_to_file('data/lattices/test_lattice.txt')
    LS.save_profile_to_file('data/lattices/profile_test.txt', ion)

    alpha = np.arange(-180,180,10)*np.pi/180.0
    dW = 0.007*0.504e6
    dPhi = 10
    khi = -0.9



    particles=[]
    for i,a in enumerate(alpha):
        p = Particle(238, 33, 0.504e6+dW*(np.cos(a)*np.sin(khi)+np.sin(a)*np.cos(khi)), initial_phase=dPhi*np.cos(a))
        particles.append(p)

    t0 = time.time()
    pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
    results = pool.map(LS.track_particle, particles)

    for res in results:
        zp = res[0]
        taup = res[1]
        Wp = res[2]
        plt.plot(zp, (taup-ion.taup)*180/np.pi,c='C0',linewidth=0.3)
        #plt.plot(p.zp, (p.Wp - ion.Wp)/ ion.Wp)
    plt.bar(LS.cavities.zc, np.ones(len(LS.cavities)), width=0.2, color='k')
    plt.rcParams['figure.figsize']=[12,4]
    t1 = time.time()
    print('Beam calculation time:', t1 - t0)
    plt.show()


    particles=[]
    for i,a in enumerate(alpha):
        p = Particle(238, 33, 0.504e6+dW*(np.cos(a)*np.sin(khi)+np.sin(a)*np.cos(khi)), initial_phase=dPhi*np.cos(a))
        particles.append(p)

    t0 = time.time()
    for p in particles:
        LS.track_particle(p)
        plt.plot(p.zp, (p.taup-ion.taup)*180/np.pi,c='C0',linewidth=0.3)
        #plt.plot(p.zp, (p.Wp - ion.Wp)/ ion.Wp)
    plt.bar(LS.cavities.zc, np.ones(len(LS.cavities)), width=0.2, color='k')
    plt.rcParams['figure.figsize']=[12,4]
    t1 = time.time()
    print('Beam calculation time:', t1 - t0)
    plt.show()


    #for p in particles:
    #    plt.plot((p.taup[0]-ion.taup[0])*180/np.pi,(p.Wp[0]-ion.Wp[0])/ion.Wp[0],'.',c='C0')
    #    plt.plot((p.taup[1]-ion.taup[1]) * 180 / np.pi, (p.Wp[1]-ion.Wp[1]) / ion.Wp[1], '.',c='C1')
    #    plt.plot((p.taup[2]-ion.taup[2]) * 180 / np.pi, (p.Wp[2]-ion.Wp[2])/ ion.Wp[2], '.', c='C2')
    #    plt.plot((p.taup[3] - ion.taup[3]) * 180 / np.pi, (p.Wp[3] - ion.Wp[3]) / ion.Wp[3], '.', c='C3')
    #plt.xlim(-20,20)
    #plt.ylim(-0.02,0.02)
    #plt.show()

    cavity_index = 5

    # LS.cavities[cavity_index].turn_on()
    # LS.cavities[cavity_index].scale = 5.1

    # print(LS.max_acceleration_phase(cavity_index, ion))
    # print(LS.get_phys(cavity_index, ion))
    # LS.set_phys(-30, cavity_index, ion)
    # print(LS.cavities[cavity_index].phase * 180 / np.pi, LS.get_phys(cavity_index, ion))

    test_phase_scan = 0
    if test_phase_scan:
        t0 = time.time()
        phases = np.arange(-180, 180.1, 30)
        W = []
        bpm_phase = []
        for phase in phases:
            LS.cavities[cavity_index].phase = phase * np.pi / 180.0
            LS.track_particle(ion)
            W.append(ion.Wp[-1] / ion.A / 1e6)
            bpm_phase.append(LS.bpms[-1].phase - LS.bpms[-2].phase)
        t1 = time.time()
        print('Phase scan of cavity: ', LS.cavities[cavity_index].name)
        print('Original tracking scheme:', t1 - t0)
        # print(W.index(max(W)))
        # print(bpm_phase.index(min(bpm_phase)))

        plt.plot(phases, W, '.-')
        # plt.plot(phases, -np.array(bpm_phase)*0.02+1.51, '.-')
        plt.show()

        element_index = LS.elements.index(LS.cavities[cavity_index])
        t0 = time.time()
        phases = np.arange(-180, 180.1, 10)
        W = []
        bpm_phase = []
        for phase in phases:
            LS.cavities[cavity_index].phase = phase * np.pi / 180.0
            LS.track_particle(ion, n_start=element_index - 1)
            W.append(ion.Wp[-1] / ion.A / 1e6)
            bpm_phase.append(LS.bpms[-1].phase - LS.bpms[-2].phase)
        t1 = time.time()
        print('New tracking scheme:', t1 - t0)

        plt.plot(phases, W, '.-')
        plt.show()




    # print(LS.bpms.phase)

    # plt.plot(LS.bpms.zc, LS.bpms.phase, '.-')
    # plt.plot(LS.cavities.zc, LS.cavities.on_off_status, '.-')
    # plt.show()

    # LS.save_lattice_to_file('data/lattices/test.txt')
    # LS.save_profile_to_file('data/lattices/profile_test.txt', ion)
