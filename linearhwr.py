from __future__ import annotations
from abc import abstractmethod as abstract, ABC as Abstract
from typing import Callable, Union, Any, cast
from math import cos, pi
from sympy import symbols, Symbol, Add, nsolve, Eq, Expr, Piecewise, exp  # pyright: ignore
from sympy.core.relational import Relational
from sympy.matrices import Matrix

number = Union[float, int]

class Component(Abstract):
    TERMNUM: int
    CURRNUM: int

    def __init__(self, name: str):
        self.name: str = name
        self.terminals: dict[str, int] = {}
        self.id: Union[int, None] = None
        self.settings: dict[str, Any] = {}
        self.potentials: list[Symbol] = [Symbol("undefined")] * self.TERMNUM
        self.currents: list[Symbol] = []
        self.aux: list[Symbol] = []
    
    def set_id(self, i: int):
        self.id = i
    
    def init_aux_vars(self):
        return self.aux

    @abstract
    def define(self, t: number) -> list[Relational] | Relational:
        pass

    @abstract
    def terminal_currents(self, t: int) -> list[number]:
        pass

class Resistor(Component):
    TERMNUM = 2
    CURRNUM = 1
    T0 = 0
    T1 = 1

    def resistance(self, ohm: number):
        self.settings["resistance"] = ohm
        return self
    
    def define(self, t: number) -> list[Relational] | Relational:
        return cast(Relational, Eq(self.potentials[Resistor.T0], self.potentials[Resistor.T1] + self.currents[0] * self.settings["resistance"]))
    
    def terminal_currents(self, t: int) -> list[number]:
        match t:
            case Resistor.T0: return [-1]
            case Resistor.T1: return [1]
            case _: raise ValueError()

class VoltageSource(Component):
    TERMNUM = 2
    CURRNUM = 1
    PLUS = 0
    MINUS = 1

    def voltage(self, volt: Union[Callable[[number], number], number]):
        if isinstance(volt, number):
            self.settings["voltage"] = lambda t: volt  # pyright: ignore
        else:
            self.settings["voltage"] = volt
        return self

    def define(self, t: number):
        return cast(Relational, Eq(self.potentials[VoltageSource.PLUS], self.potentials[VoltageSource.MINUS] + self.settings["voltage"](t)))
    
    def terminal_currents(self, t: int) -> list[number]:
        match t:
            case VoltageSource.PLUS: return [1]
            case VoltageSource.MINUS: return [-1]
            case _: raise ValueError()

class CurrentSource(Component):
    TERMNUM = 2
    CURRNUM = 1
    PLUS = 0
    MINUS = 1

    def current(self, curr: Union[Callable[[number], number], number]):
        if isinstance(curr, number):
            self.settings["current"] = lambda t: curr  # pyright: ignore
        else:
            self.settings["current"] = curr
        return self

    def define(self, t: number):
        return cast(Relational, Eq(self.currents[0], self.settings["current"](t)))

    def terminal_currents(self, t: int) -> list[number]:
        match t:
            case CurrentSource.PLUS: return [1]
            case CurrentSource.MINUS: return [-1]
            case _: raise ValueError()

class Ground(Component):
    TERMNUM = 1
    CURRNUM = 1
    T = 0

    def define(self, t: number):
        return cast(Relational, Eq(self.potentials[Ground.T], 0))

    def terminal_currents(self, t: int) -> list[number]:
        return [-1]

class Diode(Component):
    TERMNUM = 2
    CURRNUM = 1
    ANODE = 0
    CATHODE = 1

    def i_s(self, v: number):
        self.settings["i_s"] = v
        return self

    def define(self, t: number):
        # Shockley Diode.
        ud: Expr = cast(Expr, self.potentials[Diode.ANODE] - self.potentials[Diode.CATHODE])
        return cast(Relational, Eq(self.currents[0], self.settings["i_s"] * (exp(ud / 0.0025) - 1)))  # pyright: ignore

    def terminal_currents(self, t: int) -> list[number]:
        match t:
            case Diode.ANODE: return [-1]
            case Diode.CATHODE: return [1]
            case _: raise ValueError()

class NMOS(Component):
    TERMNUM = 3
    CURRNUM = 2
    D = 0
    S = 1
    G = 2

    def mu0(self, v: number):
        self.settings["mu0"] = v
        return self

    def c_ox(self, v: number):
        self.settings["c_ox"] = v
        return self

    def w_l(self, v: number):
        self.settings["w/l"] = v
        return self

    def v_th(self, v: number):
        self.settings["v_th"] = v
        return self

    def modulation(self, v: number):
        self.settings["lambda"] = v
        return self

    def define(self, t: number):
        vth = self.settings["v_th"]
        uds: Expr = cast(Expr, self.potentials[NMOS.D] - self.potentials[NMOS.S])
        ugs: Expr = cast(Expr, self.potentials[NMOS.G] - self.potentials[NMOS.S])
        const = self.settings["mu0"] * self.settings["c_ox"] * self.settings["w/l"]
        return [
            cast(Relational, Eq(self.currents[0],
                                Piecewise(
                                    (0, ugs <= vth),
                                    (const * ((ugs - vth) * uds - uds**2 / 2), uds < ugs - vth),
                                    (0.5 * const * (ugs - vth)**2 * (1 + self.settings["lambda"] * (uds - ugs + vth)), True)
                                )
                               )),
            cast(Relational, Eq(self.currents[1], 0))
        ]

    def terminal_currents(self, t: int) -> list[number]:
        match t:
            case NMOS.D: return [-1, 0]
            case NMOS.S: return [1, 0]
            case NMOS.G: return [0, 1]
            case _: raise ValueError()

class PMOS(Component):
    TERMNUM = 3
    CURRNUM = 2
    D = 0
    S = 1
    G = 2

    def mu0(self, v: number):
        self.settings["mu0"] = v
        return self

    def c_ox(self, v: number):
        self.settings["c_ox"] = v
        return self

    def w_l(self, v: number):
        self.settings["w/l"] = v
        return self

    def v_th(self, v: number):
        self.settings["v_th"] = v
        return self

    def modulation(self, v: number):
        self.settings["lambda"] = v
        return self

    def define(self, t: number):
        vth = self.settings["v_th"]
        uds: Expr = cast(Expr, self.potentials[NMOS.D] - self.potentials[NMOS.S])
        ugs: Expr = cast(Expr, self.potentials[NMOS.G] - self.potentials[NMOS.S])
        const = self.settings["mu0"] * self.settings["c_ox"] * self.settings["w/l"]
        return [
            cast(Relational, Eq(-self.currents[0],
                                Piecewise(
                                    (0, ugs >= vth),
                                    (const * ((ugs - vth) * uds - uds**2 / 2), uds > ugs - vth),
                                    (0.5 * const * (ugs - vth)**2 * (1 + self.settings["lambda"] * (uds - ugs + vth)), True)
                                )
                               )),
            cast(Relational, Eq(self.currents[1], 0))
        ]

    def terminal_currents(self, t: int) -> list[number]:
        match t:
            case NMOS.D: return [-1, 0]
            case NMOS.S: return [1, 0]
            case NMOS.G: return [0, 1]
            case _: raise ValueError()

def build(t: number, short: list[list[tuple[Component, int]]], *objects: Component) -> dict[Any, Any]:
    potentials = symbols(f"φ(1:{len(short) + 1})", real=True)
    currents = symbols(f"I(1:{sum(map(lambda cmp: cmp.CURRNUM, objects)) + 1})", real=True)
    aux_vars: list[Symbol] = []
    c = 0
    for object in objects:
        aux_vars.extend(object.init_aux_vars())
        object.currents = currents[c:c + object.CURRNUM]
        c += object.CURRNUM
    syms = potentials + tuple(currents) + tuple(aux_vars)
    equations: list[Relational] = []
    for var, s in enumerate(short):
        node_equation: list[Expr] = []
        for cmp, term in s:
            cmp.potentials[term] = potentials[var]
            cmp_currents = cmp.terminal_currents(term)
            for k in range(cmp.CURRNUM):
                node_equation.append(cast(Expr, cmp_currents[k] * cmp.currents[k]))
        equations.append(cast(Relational, Eq(Add(*node_equation), 0)))
    for object in objects:
        invs = object.define(t)
        if isinstance(invs, list):
            equations.extend(invs)
        else:
            equations.append(invs)
    m: Matrix = cast(Matrix, nsolve(equations, syms, (0,) * (len(potentials) + len(currents) + len(aux_vars))))
    return {syms[k]: m[k] for k in range(len(syms))}

if __name__ == "__main__":
    print("SELECT DEMO\n(1) Voltage source and resistance\n(2) NMOS linear\n(3) PMOS / RTKS 4.1\n(4) Diode w/ alternating source")
    sel = input("> ")
    if sel == "1":
        Uq = VoltageSource("U").voltage(5)
        R1 = Resistor("R1").resistance(2)
        R2 = Resistor("R2").resistance(10)
        R3 = Resistor("R3").resistance(8)
        GND = Ground("GND")

        wires = [
            [ (Uq, VoltageSource.PLUS), (R1, Resistor.T0) ],
            [ (R1, Resistor.T1), (R2, Resistor.T0), (R3, Resistor.T0) ],
            [ (R2, Resistor.T1), (R3, Resistor.T1), (GND, Ground.T), (Uq, VoltageSource.MINUS) ]
        ]

        s = build(0, wires, Uq, R1, R2, R3, GND)  # pyright: ignore
        print("I1: %.3gA, I2: %.3gA, I3: %.3gA" % (s[R1.currents[0]], s[R2.currents[0]], s[R3.currents[0]]))
    elif sel == "2":
        Uq1 = VoltageSource("Uq1").voltage(3)
        Uq2 = VoltageSource("Uq2").voltage(5)
        T = NMOS("T").mu0(1).c_ox(1).w_l(1).v_th(1).modulation(0.19)
        GND = Ground("GND")

        wires = [
            [ (Uq1, VoltageSource.MINUS), (Uq2, VoltageSource.MINUS), (T, NMOS.S), (GND, Ground.T) ],
            [ (Uq1, VoltageSource.PLUS), (T, NMOS.D) ],
            [ (Uq2, VoltageSource.PLUS), (T, NMOS.G) ]
        ]

        s = build(0, wires, Uq1, Uq2, T, GND)  # pyright: ignore
        print("I_DS = %.3gA" % s[T.currents[0]])
    elif sel == "3":
        Uq1 = VoltageSource("Uq1").voltage(3)
        UG = VoltageSource("UG").voltage(1)
        UD = VoltageSource("UD").voltage(2)
        T = PMOS("T").mu0(250e-4).c_ox(0.34e-10/3.3e-9).w_l(25).v_th(-0.5).modulation(0.19)
        GND = Ground("GND")

        wires = [
            [ (Uq1, VoltageSource.MINUS), (UG, VoltageSource.MINUS), (UD, VoltageSource.MINUS), (GND, Ground.T) ],
            [ (UD, VoltageSource.PLUS), (T, PMOS.D) ],
            [ (Uq1, VoltageSource.PLUS), (T, PMOS.S) ],
            [ (UG, VoltageSource.PLUS), (T, PMOS.G) ]
        ]

        s = build(0, wires, Uq1, UG, UD, T, GND)  # pyright: ignore
        print("I_D = %.3gmA" % (-s[T.currents[0]] * 1000))
    elif sel == "4":
        Uq = VoltageSource("Uq").voltage(lambda t: 5 * cos(2 * pi * t))
        R = Resistor("R").resistance(1000)
        D = Diode("D").i_s(10e-21)
        GND = Ground("GND")

        wires = [
            [ (Uq, VoltageSource.MINUS), (D, Diode.CATHODE), (GND, Ground.T) ],
            [ (Uq, VoltageSource.PLUS), (R, Resistor.T0) ],
            [ (R, Resistor.T1), (D, Diode.ANODE) ]
        ]

        s0 = build(0, wires, Uq, R, D, GND)  # pyright: ignore
        s1 = build(0.5, wires, Uq, R, D, GND)  # pyright: ignore

        print("I(0) = %.3gmA\nI(0.5s) = %.2eA" % (s0[D.currents[0]] * 1000, s1[D.currents[0]]))
