class Element:
    def __init__(self, name, zc=0.0, offset=0.0, phase=0.0):
        self.name = name
        self.zc = zc
        self.phase = phase
        self.offset = offset
        self.beam_beta = 0.0

    def __repr__(self):
        return self.name + ' ' + str(self.phase)


class BPM(Element):
    def __init__(self, name, zc=0.0, offset=0.0, phase=0.0):
        super().__init__(name, zc, offset, phase)
        self.scale = 0
        self.lo = 0.0
        self.lolo = 0.0
        self.hi = 0.0
        self.hihi = 0.0

class Stripper(Element):
    def __init__(self, name, zc=0.0, offset=0.0, phase=0.0):
        super().__init__(name, zc, offset, phase)
        self.scale = 0
        self.lo = 0.0
        self.lolo = 0.0
        self.hi = 0.0
        self.hihi = 0.0

class Cavity(Element):
    def __init__(self, name, model, zc=0.0, offset=0.0, scale=0.0, phase=0.0):
        super().__init__(name, zc, offset, phase)
        self._model = model
        self._on = True
        self.length = self._model.length
        self.frequency = self._model.frequency
        self.type = self._model.fieldmap_filename
        self.scale = scale
        self.is_on = 1
        self.is_off = 0
        self.field_amplitude = 0.0
        self.synchronous_phase = 0.0
        self.lo = 0.5
        self.lolo = 0.0
        self.hi = 8.0
        self.hihi = 8.0

    @property
    def on_off_status(self):
        return self._on

    def turn_on(self):
        self._on = True

    def turn_off(self):
        self._on = False

    def field(self, z):
        return self._model.field(z) * self._on

    def accelerating_voltage(self, beta):
        return self._model.accelerating_voltage(beta)

    def T(self, beta):
        return self._model.T(beta)

    def T2(self, beta):
        return self._model.T2(beta)

    def T2s(self, beta):
        return self._model.T2s(beta)

    def T_prime(self, beta):
        return self._model.T_prime(beta)


class ElementContainer(list):
    def __getattr__(self, attr):
        return [getattr(elem, attr) for elem in self]


if __name__ == '__main__':
    from cavity_model import CavityModel

    QWR041 = CavityModel('..//Ez041.txt', 80.5e6)
    cav = Cavity('test_cavity', zc=0.55, phase=0.0, offset=0.0, model=QWR041)
    print(cav.field(0.05))
    cav.turn_off()
    print(cav.field(0.05))
    cav.turn_on()
    print(cav.field(0.05), cav.length, cav.frequency, cav.type, cav.zc, cav.name)
