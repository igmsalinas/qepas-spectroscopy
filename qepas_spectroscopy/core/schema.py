"""Canonical names for the calibration array schema."""

SIGNAL_ARRAY_NAMES = (
    "modulo_remuestreado",
    "fase_remuestreada",
    "X_remuestreada",
    "Y_remuestreada",
)
SCALAR_ARRAY_NAMES = (
    "vector_med_presion",
    "vector_cons_presion",
    "vector_med_flujo",
    "vector_cons_flujo",
    "vector_temp_Vflujo",
)
REQUIRED_ARRAY_NAMES = SIGNAL_ARRAY_NAMES + SCALAR_ARRAY_NAMES
