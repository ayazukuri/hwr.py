from hwr import Resistor, VoltageSource, Ground, Diode, build
from math import sin, tau

uq = VoltageSource("uq").voltage(lambda t: 5 * sin(tau * t))
r = Resistor("r").resistance(1)
gnd = Ground("gnd")
d = Diode("d")

wires = [
    [(gnd, Ground.T), (uq, VoltageSource.MINUS), (r, Resistor.T1)],
    [(uq, VoltageSource.PLUS), (d, Diode.PLUS)],
    [(d, Diode.MINUS), (r, Resistor.T0)]
]

print(build(0.75, wires, uq, r, gnd, d))
