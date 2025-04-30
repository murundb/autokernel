from enum import Enum
from dataclasses import dataclass

@dataclass(frozen=True)
class GPUInfo:
    hardware: str
    software: str

class GPUType(Enum):
    Qualcomm = GPUInfo("Adreno", "OpenCL")
    Nvidia = GPUInfo("RTX", "Cuda")
    Apple = GPUInfo("M1", "Metal")
    OTHER = GPUInfo("Other", "Unknown")