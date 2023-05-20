from __future__ import annotations
from abc import abstractmethod as abstract, ABC as Abstract
from typing import Callable, Union, Any
import numpy as np

number = float | int

class Component(Abstract):
    TERMNUM: int
    CURRNUM: int

    def __init__(self, name: str):
        self.name: str = name
        self.terminals: dict[str, int] = {}
        self.id: int | None = None
        self.settings: dict[str, Any] = {}
        self.potentials: list[int] = [-1] * self.TERMNUM
        self.currents: list[int] = [-1] * self.CURRNUM
    
    def set_id(self, i: int):
        self.id = i

    @abstract
    def null(self, t: number) -> list[tuple[Any, list[Any], list[Any]]]:
        pass

    @abstract
    def terminal_currents(self, ter: int) -> list[number]:
        pass

class Resistor(Component):
    TERMNUM = 2
    CURRNUM = 1
    T0 = 0
    T1 = 1

    def resistance(self, ohm: Callable[[number], number] | number):
        if isinstance(ohm, number):
            self.settings["resistance"] = lambda t: ohm  # type: ignore
        else:
            self.settings["resistance"] = ohm
        return self
    
    def null(self, t: number):
        return [
            (0, [1, -1], [-self.settings["resistance"](t)])
        ]
    
    def terminal_currents(self, ter: int) -> list[number]:
        match ter:
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
            self.settings["voltage"] = lambda t: volt  # type: ignore
        else:
            self.settings["voltage"] = volt
        return self

    def null(self, t: number):
        return [
            (self.settings["voltage"](t), [1, -1], [0])
        ]
    
    def terminal_currents(self, ter: int) -> list[number]:
        match ter:
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
            self.settings["current"] = lambda t: curr  # type: ignore
        else:
            self.settings["current"] = curr
        return self

    def null(self, t: number):
        return [
            (self.settings["current"], [0, 0], [1])
        ]

    def terminal_currents(self, ter: int) -> list[number]:
        match ter:
            case CurrentSource.PLUS: return [1]
            case CurrentSource.MINUS: return [-1]
            case _: raise ValueError()

class Ground(Component):
    TERMNUM = 1
    CURRNUM = 1
    T = 0

    def null(self, t: number):
        return [
            (0, [1], [0])
        ]

    def terminal_currents(self, ter: number) -> list[number]:
        return [-1]

"""
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
"""
def build(t: number, short: list[list[tuple[Component, int]]], *objects: Component):
    n_pot = len(short)
    n_cur = sum(map(lambda o: o.CURRNUM, objects))
    n = n_pot + n_cur
    c = 0
    for object in objects:
        object.currents = [n_pot + c + k for k in range(object.CURRNUM)]
        c += object.CURRNUM
    eq: Callable[[], list[number]] = lambda : [0] * n
    equations: list[list[number]] = []
    b: list[number] = [0] * len(short)
    for potential, s in enumerate(short):
        node_equation = eq()
        for component, terminal in s:
            component.potentials[terminal] = potential
            polarization = component.terminal_currents(terminal)
            for num, cur in enumerate(component.currents):
                node_equation[cur] = polarization[num]
        equations.append(node_equation)
    for object in objects:
        for const, pots, curs in object.null(t):
            mesh_equation = eq()
            b.append(const)
            for pot, val in enumerate(pots):
                mesh_equation[object.potentials[pot]] = val
            for cur, val in enumerate(curs):
                mesh_equation[object.currents[cur]] = val
            equations.append(mesh_equation)
    return np.linalg.solve(equations, b)


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

    sol = build(0, wires, Uq, R1, R2, R3, GND)
    print(sol)
    print(sol[Uq.currents[0]])
