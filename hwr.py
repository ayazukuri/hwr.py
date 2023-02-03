from __future__ import annotations
from abc import abstractmethod as abstract, ABC as Abstract
from typing import Callable, Union, Any
from sympy import symbols, Symbol, Add, solve, Eq

number = Union[float, int]

class Component(Abstract):
    TERMNUM: int
    CURRNUM: int

    def __init__(self, name):
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
    def invariants(self, t: number) -> list[Any]:
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
    
    def invariants(self, t):
        return [
            Eq(self.potentials[Resistor.T0], self.potentials[Resistor.T1] + self.currents[0] * self.settings["resistance"])
        ]
    
    def terminal_currents(self, t):
        match t:
            case Resistor.T0: return [-1]
            case Resistor.T1: return [1]

class VoltageSource(Component):
    TERMNUM = 2
    CURRNUM = 1
    PLUS = 0
    MINUS = 1

    def voltage(self, volt: Union[Callable[[number], number], number]):
        if isinstance(volt, number):
            self.settings["voltage"] = lambda t: volt
        else:
            self.settings["voltage"] = volt
        return self

    def invariants(self, t):
        return [
            Eq(self.potentials[VoltageSource.PLUS], self.potentials[VoltageSource.MINUS] + self.settings["voltage"](t))
        ]
    
    def terminal_currents(self, t):
        match t:
            case VoltageSource.PLUS: return [1]
            case VoltageSource.MINUS: return [-1]

class CurrentSource(Component):
    TERMNUM = 2
    CURRNUM = 1
    PLUS = 0
    MINUS = 1

    def current(self, curr: Union[Callable[[number], number], number]):
        if isinstance(curr, number):
            self.settings["current"] = lambda t: curr
        else:
            self.settings["current"] = curr
        return self

    def invariants(self, t):
        return [
            Eq(self.currents[0], self.settings["current"](t))
        ]

    def terminal_currents(self, t):
        match t:
            case CurrentSource.PLUS: return [1]
            case CurrentSource.MINUS: return [-1]

class Ground(Component):
    TERMNUM = 1
    CURRNUM = 1
    T = 0

    def invariants(self, t):
        return [
            Eq(self.potentials[Ground.T], 0)
        ]

    def terminal_currents(self, t):
        return [-1]

def build(t: number, short: list[list[tuple[Component, int]]], *objects: Component):
    potentials = symbols(f"φ(1:{len(short) + 1})", real=True)
    currents = symbols(f"I(1:{sum(map(lambda cmp: cmp.CURRNUM, objects)) + 1})", real=True)
    aux_vars = []
    c = 0
    for object in objects:
        aux_vars.extend(object.init_aux_vars())
        object.currents = currents[c:c + object.CURRNUM]
        c += 1
    equations = []
    for var, s in enumerate(short):
        node_equation = []
        for cmp, term in s:
            cmp.potentials[term] = potentials[var]
            cmp_currents = cmp.terminal_currents(term)
            for k in range(cmp.CURRNUM):
                node_equation.append(cmp_currents[k] * cmp.currents[k])
        equations.append(Add(*node_equation))
    for object in objects:
        equations.extend(object.invariants(t))
    return solve(equations, potentials + tuple(currents) + tuple(aux_vars))

if __name__ == "__main__":
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

    sol: list[number] = list(build(0, wires, Uq, R1, R2, R3, GND).values())
    print("U_Q: %.3f" % (sol[0] - sol[2]))
    print("U_1: %.3f" % (sol[0] - sol[1]))
    print("U_23: %.3f" % (sol[1] - sol[2]))
    print("I: %.3f" % sol[3])
