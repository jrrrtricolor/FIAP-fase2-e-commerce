"""Código específico do recomendador do Tech Challenge Fase 2."""

import os

# Evitar warning do joblib ao contar CPUs físicas no macOS.
limite_cpu_joblib = max(1, (os.cpu_count() or 2) - 1)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(limite_cpu_joblib))
