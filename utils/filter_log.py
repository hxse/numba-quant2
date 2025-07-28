import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

# 隐藏 Numba 的 First-class function type experimental warning
warnings.filterwarnings(
    "ignore",
    category=NumbaExperimentalFeatureWarning,
    message="First-class function type feature is experimental")
