# -*- coding: utf-8 -*-
"""Orquestrador de export para fechamento da entrega (TCC)."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from results_entrega import run_export_entrega

if __name__ == '__main__':
    run_export_entrega()
