"""Configuration loader and parser for model-car YAML files.

Handles:
1. YAML parsing and validation
2. Model requirement extraction
3. Accelerator configuration management
"""

import json
import logging
import math
import subprocess
from typing import Dict
import re

import yaml


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_llm_model_config(config_path: str) -> dict:
    """Load and parse the model-car YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.
    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def _skopeo_inspect(image_name: str) -> dict | None:
    image_ref = image_name.split("://", 1)[1] if "://" in image_name else image_name
    image_ref = f"docker://{image_ref}"
    try:
        result = subprocess.run(
            ["skopeo", "inspect", "--override-os", "linux", "--override-arch", "amd64", image_ref],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        logger.error("skopeo inspect failed for %s: %s", image_name, exc)
        return None

def _estimate_model_size(image_name: str) -> int:
    """Estimate the model size based on the image name using podman inspect.

    Args:
        image_name (str): The name of the container image.
    Returns:
        int: Size of the image in bytes.
    """
    metadata = _skopeo_inspect(image_name)

    if not metadata:
        return 0
    layers = metadata.get("LayersData", [])
    image_size_bytes = sum(layer.get("Size", 0) for layer in layers if isinstance(layer, dict))
    if not image_size_bytes:
        return 0

    size_gb = image_size_bytes / (1024 ** 3)
    return math.ceil(size_gb * 100) / 100
    
def _supported_arch(image_name: str) -> str:
    """Estimate the supported architecture based on the image name using podman inspect.

    Args:
        image_name (str): The name of the container image.
    Returns:
        str: Supported architecture of the image.
    """
    metadata = _skopeo_inspect(image_name)

    if not metadata:
        return "unknown"
    arch = metadata.get("Architecture") or metadata.get("Labels", {}).get("architecture")
    return arch or "unknown"


def _parameter_count_from_name(model_name: str) -> float | None:
    """
    Return parameter scale in billions if the name contains tokens like 8b, 70b, 7.1b.
    Examples handled:
      - oci://.../modelcar-granite-3-1-8b-instruct:1.5 -> 8.0
      - mistral-7b-instruct -> 7.0

    :param model_name: The model name string.
    :return: Parameter count in billions or None if not found.

    """
    # Drop transport prefix and tag/digest
    name = model_name.split("://", 1)[-1]
    name = name.split(":", 1)[0]

    match = re.search(r"(\d+(?:\.\d+)?)\s*b", name, re.IGNORECASE)
    return float(match.group(1)) if match else None

def _quantization_bits_from_name(model_name: str) -> int | None:
    """
    Return quantization bits if the name contains tokens like int8, fp16, fp32.
    Examples handled:
      - oci://.../modelcar-granite-3-1-8b-instruct-int8:1.5 -> 8
      - modelname-fp16 -> 16

    :param model_name: The model name string.
    :return: Quantization bits or None if not found.
    """
    n = model_name.lower()

     # w4a16 / w8a8 style: return weight bits
    match = re.search(r"w(\d+)a\d+", n)
    if match:
        return int(match.group(1))

    # fp8 / fp16 / fp32
    match = re.search(r"fp(\d+)", n)
    if match:
        return int(match.group(1))

    # gptq-4bit, 4bit
    match = re.search(r"(\d+)\s*bit", n)
    if match:
        return int(match.group(1))

    return None  # fall back to default elsewhe

def _estimate_required_vram_gb(
    params_billion: float | None,
    quant_bits: int | None,
    overhead_factor: float = 0.1,
) -> int | None:
    """Estimate VRAM footprint in GB using parameter count and quantization bits."""
    if params_billion is None:
        return None
    q = quant_bits or 16
    if q <= 0:
        return None
    vram_gb = params_billion * (q / 8.0) * (1.0 + overhead_factor)
    return math.ceil(vram_gb)


def estimate_model_size(image_name: str) -> int:
    """Public wrapper to estimate model size bytes for a container image."""
    return _estimate_model_size(image_name)
    
def get_model_requirements(config: Dict) -> Dict:
    """Extract model requirements from the configuration.

    Args:
        config (Dict): Parsed configuration dictionary.
    Returns:
        Dict: Model requirements including model name and accelerator info.
    """
    model_info = config.get('model-car', [])
    if isinstance(model_info, dict):
        models = [model_info]
    else:
        models = [m for m in model_info if isinstance(m, dict)]
    
    requirements = {}
    for model_info in models:
        name = model_info.get('name', 'unknown')
        params_billion = _parameter_count_from_name(model_info.get('name', ''))
        quant_bits = _quantization_bits_from_name(model_info.get('name', ''))
        requirements[name] = {
            'model_name': model_info.get('name', 'unknown'),
            'image': model_info.get('image', 'default-image'),
            'arguments': model_info.get('serving_arguments', {}).get('args', []),
            'model_size_gb': _estimate_model_size(model_info.get('image', 'default-image')),
            'model_p_billion': params_billion,
            'quantization_bits': quant_bits,
            'required_vram_gb': _estimate_required_vram_gb(params_billion, quant_bits),
            'supported_arch': _supported_arch(model_info.get('image', 'default-image'))
        }

    return requirements
    

def calculate_gpu_requirements(
        requirements: Dict,
        overhead_factor: float = 0.1,
    ) -> int:
    """Calculate total GPU memory requirements from model requirements.

    Formula:

    Memory (GB) = P_billion * (Q / 8) * (1 + overhead_factor)

    Q - Quantization factor (8 for int8, 16 for fp16, 32 for fp32)
    P_billion - Number of parameters (in billions)
    overhead_factor - 0.1 (10% overhead)


    https://bentoml.com/llm/getting-started/calculating-gpu-memory-for-llms

    Args:
        requirements (Dict): Model requirements dictionary.
    Returns:
        int: GPU memory required
    """
    total_gb = 0.0
    for info in requirements.values():
        P = info.get("model_p_billion")
        Q = info.get("quantization_bits") or 16  # default to fp16
        if P is None:
            continue
        total_gb += P * (Q / 8.0) * (1.0 + overhead_factor)
    return math.ceil(total_gb)

def extract_deployment_matrix(matrix_path: str) -> Dict:
    """Extract deployment matrix from JSON file.

    Args:
        matrix_path (str): Path to the deployment matrix JSON file.
    Returns:
        Dict: Deployment matrix as a dictionary.
    """
    with open(matrix_path, 'r') as file:
        matrix = json.load(file)
    return matrix