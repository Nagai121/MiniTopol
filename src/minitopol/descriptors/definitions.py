"""Explicit, immutable scientific calculation definitions."""

from dataclasses import dataclass, field
import json

from ..storage import canonical, digest, identifier


@dataclass(frozen=True)
class DescriptorDefinition:
    name: str
    version: str
    representation: str
    method: str
    unit: str
    parameters: dict = field(default_factory=dict)
    _snapshot: str = field(init=False, repr=False)

    def __post_init__(self):
        identifier(self.name)
        identifier(self.representation)
        if not self.version or not self.method or not self.unit:
            raise ValueError("Version, scientific method and unit must be specified")
        object.__setattr__(self, "_snapshot", canonical({
            "name": self.name, "version": self.version,
            "representation": self.representation, "method": self.method,
            "unit": self.unit, "parameters": self.parameters}))

    def as_dict(self):
        return json.loads(self._snapshot)

    @property
    def identity(self):
        return digest(self.as_dict())

    def register(self, workspace):
        with workspace.catalog() as con:
            con.execute("INSERT INTO descriptor_definitions VALUES (?, ?) ON CONFLICT DO NOTHING",
                        [self.identity, canonical(self.as_dict())])
        return self.identity
