"""QA Specialist for running ODH validation tests."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool
import yaml

from . import SpecialistSpec
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _extract_registry_host(image: str) -> str | None:
    """Extract registry host from an image reference (e.g., oci://registry/.../name:tag)."""
    if not image:
        return None
    trimmed = image.strip()
    for prefix in ("oci://", "docker://", "http://", "https://"):
        if trimmed.startswith(prefix):
            trimmed = trimmed[len(prefix):]
            break
    if not trimmed:
        return None
    # Host is the first path segment.
    parts = trimmed.split("/", 1)
    host = parts[0].strip()
    return host or None


def _infer_registry_from_modelcar(modelcar_cfg: dict) -> str | None:
    """Infer a single registry host from model-car image fields, if possible."""
    if not isinstance(modelcar_cfg, dict):
        return None
    model_block = modelcar_cfg.get("model-car")
    if isinstance(model_block, dict):
        model_entries = [model_block]
    elif isinstance(model_block, list):
        model_entries = model_block
    else:
        return None

    registries: set[str] = set()
    for entry in model_entries:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if not isinstance(image, str):
            continue
        host = _extract_registry_host(image)
        if host:
            registries.add(host)

    if len(registries) == 1:
        return next(iter(registries))
    return None


def build_qa_specialist(
    llm: BaseChatModel,
    extract_text: Callable[[dict], str],
    precomputed_requirements: dict | None = None,
) -> SpecialistSpec:
    """Return the QA specialist agent and the supervisor-facing tool."""

    @tool
    def run_odh_tests(
        runtime_image: str,
        gpu_provider: str,
    ) -> str:
        """
        Run the ODH model validation test suite using a staged kubeconfig under /tmp.
        :param runtime_image: The vLLM runtime image to use for testing.
        """

        image = "quay.io/opendatahub/opendatahub-tests:latest"
        repo_root = Path.cwd()
        generated_config = repo_root / "config-yaml" / "sample_modelcar_config.generated.yaml"
        host_modelcar_path = generated_config
        if not host_modelcar_path.exists():
            fallback_config = repo_root / "config-yaml" / "sample_modelcar_config.base.yaml"
            if fallback_config.exists():
                print(
                    f"[QA] Generated config not found at {generated_config}. "
                    f"Falling back to base config: {fallback_config}",
                    flush=True,
                )
                host_modelcar_path = fallback_config
            else:
                return f"QA_ERROR:MODELCAR_NOT_FOUND {generated_config}"
        REGISTRY_PULL_SECRET = os.environ.get("OCI_REGISTRY_PULL_SECRET", "")
        if not REGISTRY_PULL_SECRET:
            msg = "QA_ERROR:OCI_PULL_SECRET_MISSING OCI registry pull secret not set in environment."
            logger.error(msg)
            print(f"[QA] {msg}", flush=True)
            return msg
        VLLM_RUNTIME_IMAGE = os.environ.get("VLLM_RUNTIME_IMAGE", runtime_image)

        host_kubeconfig = os.environ.get(
            "KUBECONFIG", os.path.expanduser("~/.kube/config")
        )
        host_kubeconfig_path = Path(host_kubeconfig)

        if not host_kubeconfig_path.exists():
            msg = f"QA_ERROR:KUBECONFIG_MISSING Host kubeconfig not found at {host_kubeconfig}"
            logger.error(msg)
            print(f"[QA] {msg}", flush=True)
            return msg

        tmp_dir = Path(tempfile.mkdtemp(prefix="odh-tests-"))
        staged_kubeconfig = tmp_dir / "kubeconfig"
        results_dir = tmp_dir / "results"

        if not host_modelcar_path.exists():
            return f"QA_ERROR:MODELCAR_NOT_FOUND {host_modelcar_path}"

        tmp_modelcar_path = tmp_dir / "modelcar.yaml"
        shutil.copy2(host_modelcar_path, tmp_modelcar_path)

        try:
            with open(tmp_modelcar_path, "r") as f:
                modelcar_cfg = yaml.safe_load(f)
        except Exception as e:
            msg = f"QA_ERROR:MODELCAR_YAML_INVALID Failed to parse {tmp_modelcar_path}: {e}"
            logger.error(msg)
            print(f"[QA] {msg}", flush=True)
            return msg

        registry_from_modelcar = _infer_registry_from_modelcar(modelcar_cfg or {})
        if not registry_from_modelcar:
            msg = "QA_ERROR:MODELCAR_REGISTRY_UNDETERMINED Could not determine a single registry host from model-car config."
            logger.error(msg)
            print(f"[QA] {msg}", flush=True)
            return msg
        try:
            shutil.copy2(host_kubeconfig_path, staged_kubeconfig)
            staged_kubeconfig.chmod(0o644)

            results_dir.mkdir(parents=True, exist_ok=True)
            results_dir.chmod(0o777)

            cmd = [
                "podman", "run", "--rm",
                "-e", "KUBECONFIG=/home/odh/.kube/config",
                "-e", f"OCI_REGISTRY_PULL_SECRET={REGISTRY_PULL_SECRET}",
                "-v", f"{staged_kubeconfig}:/home/odh/.kube/config:Z",
                "-v", f"{results_dir}:/home/odh/opendatahub-tests/results:Z",
                "-v", f"{tmp_modelcar_path}:/home/odh/opendatahub-tests/modelcar.yaml:Z",
                image,
                "-vv",
                "tests/model_serving/model_runtime/model_validation/test_modelvalidation.py",
                "--model_car_yaml_path=/home/odh/opendatahub-tests/modelcar.yaml",
                f"--vllm-runtime-image={VLLM_RUNTIME_IMAGE}",
                f"--supported-accelerator-type={gpu_provider}",
                f"--registry-host={registry_from_modelcar}",
                "--snapshot-update",
                "--log-file=/home/odh/opendatahub-tests/results/pytest-logs.log",
            ]


            logger.info("Running ODH tests with command: %s", " ".join(map(str, cmd)))
            print("[QA] Starting ODH tests in container...", flush=True)

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            output_lines: list[str] = []
            start = time.time()
            timeout = 1800  # 30 minutes

            assert proc.stdout is not None
            for line in proc.stdout:
                output_lines.append(line)
                print(f"[QA] {line}", end="", flush=True)

                if time.time() - start > timeout:
                    proc.kill()
                    msg = (
                        f"QA_ERROR:TIMEOUT QA test suite did not complete within "
                        f"{timeout} seconds."
                    )
                    logger.error(msg)
                    print(f"\n[QA] {msg}\n", flush=True)
                    return msg

            proc.wait()
            full_output = "".join(output_lines)

            # 5. Classify result for the supervisor / decision layer
            if proc.returncode != 0:
                logger.error("ODH tests exited with code %s", proc.returncode)

                if "Invalid kube-config file" in full_output or "No configuration found" in full_output:
                    return "QA_ERROR:KUBECONFIG_INVALID " + full_output
                if "Trying to get client via new_client_from_config" in full_output:
                    return "QA_ERROR:CLUSTER_UNREACHABLE " + full_output

                return "QA_ERROR:TESTS_FAILED " + full_output

            print("[QA] ODH tests completed successfully.\n", flush=True)
            return "QA_OK:" + full_output

        except FileNotFoundError as e:
            logger.exception("podman not found when running ODH tests")
            msg = f"QA_ERROR:RUNTIME_NOT_FOUND podman not found or not executable: {e}"
            print(f"[QA] {msg}", flush=True)
            return msg
        except Exception as e:
            logger.exception("Unexpected error while running ODH tests")
            msg = f"QA_ERROR:UNEXPECTED {e}"
            print(f"[QA] {msg}", flush=True)
            return msg

    prompt = (
        "You are a QA Specialist responsible for validating machine learning model deployments "
        "and configurations on OpenShift / Kubernetes.\n\n"
        "You have access to a tool called `run_odh_tests` which runs the Opendatahub model "
        "validation test suite inside a container, and streams logs to the console.\n\n"
        "When a user asks to validate a deployment, or when you are invoked by the supervisor:\n"
        "1. Before running the ODH test suite, the tool will automatically check for the "
        "   'raw-model-validation' namespace. If it exists, it will be deleted using "
        "   'oc delete ns raw-model-validation --force' to ensure a clean test environment.\n"
        "2. Call `run_odh_tests`.\n"
        "3. Inspect its output string.\n"
        "4. Summarize whether QA passed or failed, and why.\n"
        "5. Provide clear, concise recommendations for next steps (e.g., fix kubeconfig, fix cluster access, "
        "   investigate failing tests, etc.).\n\n"
        "Never request kubeconfig contents or secrets from the user. Work only with the logs and status provided "
        "by the tool. \n"
        "When you call 'run_odh_tests', you MUST provide the vLLM runtime image to test as the argument. \n"
        "The vllm runtime image will be provided by the supervisor agent in your input request.\n"
    )

    agent = create_agent(
        llm,
        tools=[run_odh_tests],
        system_prompt=prompt,
    )

    @tool
    def analyze_qa_results(request: str, runtime_image: str, gpu_provider: str):
        """
        Supervisor-facing entrypoint. The supervisor must pass:
            - request: what to do (e.g. "run QA and summarize results")
            - runtime_image: the vLLM runtime image to test
            - gpu_provider: the GPU provider (e.g. "NVIDIA" or "AMD")
        """
        qa_input = (
            f"{request}\n\n"
            f"RUNTIME_IMAGE::{runtime_image}\n"
            f"GPU_PROVIDER::{gpu_provider}\n"
            "You MUST call `run_odh_tests` using this exact runtime image."
        )

        result = agent.invoke({"messages": [{"role": "user", "content": qa_input}]})
        return extract_text(result)


    analyze_qa_results.name = "analyze_qa_results"

    return SpecialistSpec(
        name="qa_specialist",
        agent=agent,
        tool=analyze_qa_results,
    )


__all__ = ["build_qa_specialist"]
