"""Configuration specialist agent."""

from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
import yaml
from ...config.model_config import calculate_gpu_requirements, extract_deployment_matrix
from ...utils.path_utils import detect_repo_root

from . import SpecialistSpec


def build_config_specialist(
    llm: BaseChatModel,
    extract_text: Callable[[dict], str],
    precomputed_requirements: dict | None = None,
    bootstrap_config_path: str | None = None,
) -> SpecialistSpec:
    """Return the configuration specialist agent and the supervisor-facing tool."""

    @tool
    def describe_preloaded_requirements() -> str:
        """Return the preloaded model requirements as JSON."""
        return json.dumps(precomputed_requirements, indent=2)
    
    @tool
    def infer_gpu_needs() -> str:
        """Infer VRAM needs from preloaded requirements."""
        if not precomputed_requirements:
            return "No preloaded requirements available to infer VRAM needs."
        total_vram = calculate_gpu_requirements(precomputed_requirements)
        per_model = [
            f"{name}: {info.get('required_vram_gb', 'unknown')} GB"
            for name, info in precomputed_requirements.items()
        ]
        per_model_str = "; ".join(per_model)
        return f"Total VRAM requirements inferred: {total_vram} GB; per-model: {per_model_str}"
    
    @tool
    def generate_optimal_serving_arguments(optimized_args_json: str) -> str:
        """
        Create a *new* model-car YAML that only includes deployable models and applies
        optimized serving arguments for those models.

        Inputs:
        - optimized_args_json:
            EITHER a single object:
                {
                "model_name": "granite-3.1-8b-instruct",
                "serving_arguments": { ... }
                }
            OR a list of such objects:
                [
                {
                    "model_name": "...",
                    "serving_arguments": { ... }
                },
                ...
                ]

        Behavior:
        - Reads the base model-car config from:
            - bootstrap_config_path (if provided), otherwise
            - <repo_root>/config-yaml/sample_modelcar_config.base.yaml
        - Reads deployability decisions from:
            <repo_root>/info/deployment_matrix.json
            Each entry should look like:
            {
                "model_name": "granite-3.1-8b-instruct",
                "deployable": true,
                "reason": "..."
            }
        - Keeps ONLY models with deployable == true.
        - Applies serving_arguments overrides from optimized_args_json
            to those deployable models (if present).
        - Writes the final filtered config to:
            <repo_root>/config-yaml/sample_modelcar_config.generated.yaml
        """

        # ------------------------------------------------------------------ #
        # 1. Resolve paths
        # ------------------------------------------------------------------ #
        repo_root = detect_repo_root()
        output_path = Path(repo_root, "config-yaml", "sample_modelcar_config.generated.yaml")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if bootstrap_config_path is not None:
            modelcar_path = Path(bootstrap_config_path)
        else:
            modelcar_path = Path(repo_root, "config-yaml", "sample_modelcar_config.base.yaml")

        if not modelcar_path.exists():
            return "Error: model-car config not found."

        deployment_matrix_path = Path(repo_root, "info", "deployment_matrix.json")

        # ------------------------------------------------------------------ #
        # 2. Load decision matrix (deployable vs non-deployable)
        # ------------------------------------------------------------------ #
        deployable_names: set[str] = set()
        if deployment_matrix_path.exists():
            try:
                with open(deployment_matrix_path, "r", encoding="utf-8") as f:
                    deployment_matrix = json.load(f)
                if isinstance(deployment_matrix, list):
                    for entry in deployment_matrix:
                        if (
                            isinstance(entry, dict)
                            and entry.get("deployable") is True
                            and isinstance(entry.get("model_name"), str)
                        ):
                            deployable_names.add(entry["model_name"])
            except Exception as e:
                # If this fails, treat it as "no decision matrix" and keep all models.
                return f"Error: Failed to read decision_matrix.json: {e}"
        else:
            return "Error: decision_matrix.json not found. Cannot determine deployable models."

        # ------------------------------------------------------------------ #
        # 3. Load base model-car YAML
        # ------------------------------------------------------------------ #
        with open(modelcar_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        model_car_block = cfg.get("model-car", [])
        if isinstance(model_car_block, dict):
            model_list = [model_car_block]
            was_dict = True
        elif isinstance(model_car_block, list):
            model_list = model_car_block
            was_dict = False
        else:
            return "Error: 'model-car' section is not a dict or list; cannot apply overrides."

        # ------------------------------------------------------------------ #
        # 4. Parse serving-argument overrides
        # ------------------------------------------------------------------ #
        try:
            overrides_obj = json.loads(optimized_args_json)
        except json.JSONDecodeError:
            return "Error: Provided serving arguments are not valid JSON."

        if isinstance(overrides_obj, dict):
            overrides_list = [overrides_obj]
        elif isinstance(overrides_obj, list):
            overrides_list = overrides_obj
        else:
            return (
                "Error: Expected a JSON object or a list of objects with "
                "'model_name' and 'serving_arguments'."
            )

        # Build a quick lookup: model_name -> serving_arguments dict
        overrides_by_name: dict[str, dict] = {}
        for ov in overrides_list:
            if not isinstance(ov, dict):
                continue
            name = ov.get("model_name")
            sa = ov.get("serving_arguments") or {}
            if isinstance(name, str) and isinstance(sa, dict) and sa:
                overrides_by_name[name] = sa

        # ------------------------------------------------------------------ #
        # 5. Filter to deployable models and apply overrides
        # ------------------------------------------------------------------ #
        new_model_list: list[dict] = []
        updated_models: list[str] = []

        for entry in model_list:
            if not isinstance(entry, dict):
                continue

            name = entry.get("name")
            if not isinstance(name, str):
                continue

            # Skip models that are NOT deployable
            if name not in deployable_names:
                continue

            # Apply overrides if present
            if name in overrides_by_name:
                if "serving_arguments" not in entry or not isinstance(entry["serving_arguments"], dict):
                    entry["serving_arguments"] = {}
                entry["serving_arguments"].update(overrides_by_name[name])
                updated_models.append(name)

            new_model_list.append(entry)

        if not new_model_list:
            return (
                "No deployable models found based on the deployment matrix. "
                "Generated config not written."
            )

        # Restore shape: dict vs list, following original config
        if was_dict and len(new_model_list) == 1:
            cfg["model-car"] = new_model_list[0]
        else:
            cfg["model-car"] = new_model_list

        # Remove any top-level serving_arguments block if present
        cfg.pop("serving_arguments", None)

        # ------------------------------------------------------------------ #
        # 6. Write generated YAML
        # ------------------------------------------------------------------ #
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                cfg,
                f,
                sort_keys=False,
                default_flow_style=False,
            )

        updated_str = ", ".join(sorted(set(updated_models))) if updated_models else "none (no overrides applied)"
        return (
            f"Generated filtered config with deployable models only at: {output_path}. "
            f"Applied serving_arguments overrides for: {updated_str}."
        )
                                                                                            

            
    prompt = """
        You are the Configuration Specialist.

        Your responsibilities:
        1. Load and analyze the preloaded model-car requirements.
        2. Infer VRAM requirements using the cached model information.
        3. Provide per-model deployment summaries.
        4. When the supervisor or Decision Specialist provides optimized serving arguments JSON,
        apply them to the model-car YAML by calling the tool:
        generate_optimal_serving_arguments(optimized_args_json).

        Rules:
        - Always begin your reasoning with a short checklist using [ ] and [x].
        - The checklist must always include:
            [ ] Load preloaded requirements
            [ ] Infer VRAM needs
            [ ] Prepare configuration summary
        (Only include YAML-update tasks if explicitly requested.)
        - For normal config analysis, you must call describe_preloaded_requirements() first,
        then infer_gpu_needs(), before writing any report.
        - Never ask the user for file paths or external input.

        Handling optimized serving arguments:
        - If the request you receive contains an 'OPTIMIZED_SERVING_ARGUMENTS_JSON' block with a JSON
        code fence, you MUST:
        1) Extract the JSON content inside the ```json ... ``` fence exactly.
        2) Call generate_optimal_serving_arguments(optimized_args_json=<that JSON string>).
        3) Return ONLY the message returned by generate_optimal_serving_arguments, without additional prose.

        Output Requirements:
        - For configuration reports, provide a clean and concise summary:
            - model names of each preloaded model
            - image size (GB)
            - parameter counts
            - quantization bits
            - estimated VRAM needs
            - supported architectures
        - When updating YAML, respond only with the return value of the tool you call.

        Tools available:
        - describe_preloaded_requirements(): returns cached model-car fields as JSON
        - infer_gpu_needs(): computes estimated VRAM from cached requirements
        - generate_optimal_serving_arguments(optimized_args_json): updates modelcar.yaml on disk

        Use the tools appropriately based on the user's request.
        """


    agent = create_agent(
        llm,
        tools=[describe_preloaded_requirements, infer_gpu_needs, generate_optimal_serving_arguments],
        system_prompt=prompt,
    )

    @tool
    def analyze_model_config(request: str) -> str:
        """Delegate configuration requests to the configuration specialist."""
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        return extract_text(result)

    analyze_model_config.name = "analyze_model_config"

    return SpecialistSpec(
        name="config_specialist",
        agent=agent,
        tool=analyze_model_config,
    )


__all__ = ["build_config_specialist"]
