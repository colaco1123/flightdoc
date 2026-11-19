"""Checks register themselves in ``REGISTRY`` when their module is imported."""

from . import control, link, motors, navigation, power, vibration
from .base import REGISTRY, Finding, Incident, Panel, Plot

__all__ = ["REGISTRY", "Finding", "Incident", "Panel", "Plot",
           "control", "link", "motors", "navigation", "power", "vibration"]
