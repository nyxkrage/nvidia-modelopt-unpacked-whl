# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Module to specify the basic hyperparameter with tracing capabilities."""

from typing import Callable, Dict, Optional, Set, Type

from modelopt.torch.opt.hparam import Hparam
from modelopt.torch.trace import Symbol

__all__ = ["TracedHpRegistry", "TracedHp"]

TracedHparamType = Type["TracedHp"]


class TracedHpRegistry:
    """A simple registry to keep track of different traced hp classes and their symbols."""

    _registry: Dict[Type[Symbol], TracedHparamType] = {}  # registered classes

    @classmethod
    def register(cls, sym_cls: Type[Symbol]) -> Callable[[TracedHparamType], TracedHparamType]:
        """Use this to register a new traced hparam class for the provided symbol class.

        Usage:

        .. code-block:: python

            @TracedHpRegistry.register(MySymbol)
            class MyHparam(TracedHp):
                ...
        """

        def decorator(hp_class: TracedHparamType) -> TracedHparamType:
            """Register hp_class with appropriate sym_class."""
            assert sym_cls not in cls._registry, f"{sym_cls} already registered!"
            cls._registry[sym_cls] = hp_class
            return hp_class

        return decorator

    @classmethod
    def unregister(cls, sym_cls: Type[Symbol]) -> None:
        """Unregister a previously registered symbol class.

        It throws a KeyError if the hparam class is not registered.
        """
        if sym_cls not in cls._registry:
            raise KeyError(f"{sym_cls} is not registered!")
        cls._registry.pop(sym_cls)

    @classmethod
    def initialize_from(cls, sym: Symbol, hp: Hparam) -> "TracedHp":
        """Initialize the sym-appropriate hparam from a vanilla hparam."""
        return cls._registry[type(sym)].initialize_from(hp)

    @classmethod
    def get(cls, sym: Symbol) -> Optional[TracedHparamType]:
        """Get Hparam type associated with symbol."""
        return cls._registry.get(type(sym))


@TracedHpRegistry.register(Symbol)
class TracedHp(Hparam):
    """A hparam that exhibits additional functionality required to handle tracing."""

    @classmethod
    def initialize_from(cls, hp: Hparam) -> "TracedHp":
        """Initialize a new hparam from an existing vanilla Trace."""
        # sanity checks
        assert type(hp) == TracedHp, f"Can only initialize from an {TracedHp} object."
        # relegate implementation to child class
        return cls._initialize_from(hp)

    @classmethod
    def _initialize_from(cls, hp: Hparam) -> "TracedHp":
        """Initialize a new hparam from an existing vanilla Hparam."""
        hp.__class__ = cls
        assert isinstance(hp, cls)
        return hp

    def resolve_dependencies(
        self, sym: Symbol, get_hp: Callable[[Symbol], "TracedHp"]
    ) -> Dict[Symbol, "TracedHp"]:
        """Resolve dependencies of the hparam via symbolic map.

        This method iterates through the dependency map described in sym and generates an
        appropriate hparam based on the currently assigned hparam and the parent symbol.

        Args:
            sym: The symbol associated with self for which we want to resolve dependencies.
            get_hp: A function that returns the hparam associated with a symbol.

        Returns:
            A mapping describing the hparam that should be associated with each symbol.
        """
        # dependency resolution only works for searchable symbols and must be associated with self
        assert sym.is_searchable or sym.is_constant, f"Symbol {sym} must be searchable or constant!"
        assert get_hp(sym) is self, f"Symbol {sym} must be associated with self to resolve!"
        assert TracedHpRegistry.get(sym) == type(self), "Symbol and Hparam type must match!"

        # relegate implementation to child class
        return self._resolve_dependencies(sym, get_hp)

    def _resolve_dependencies(
        self, sym: Symbol, get_hp: Callable[[Symbol], "TracedHp"]
    ) -> Dict[Symbol, "TracedHp"]:
        # check sortability of sym
        if not sym.is_sortable:
            self._importance_estimators = None

        # process dependencies and generate hparams
        to_process = set(sym._dependencies)
        processed: Set[Symbol] = set()
        mapping: Dict[Symbol, TracedHp] = {sym: self}

        while to_process:
            # get a new symbol
            sym_dep = to_process.pop()
            processed.add(sym_dep)

            # this should never be any other symbol
            assert type(sym_dep) == Symbol, f"Unexpected type {type(sym_dep)} for {sym_dep}!"

            # merge hparam into self and store mapping
            self &= get_hp(sym_dep)
            mapping[sym_dep] = self

            # add dependencies of sym_dep to to_process
            to_process |= set(sym_dep._dependencies) - processed

        # check constant case at the end
        if sym.is_constant:
            self.active = self.original
            self.choices = [self.original]

        return mapping
