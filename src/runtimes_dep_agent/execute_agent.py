"""Main entry point for running the supervisor agent."""

from __future__ import annotations

import argparse
import os
from langgraph.errors import GraphRecursionError
from pathlib import Path
from .utils.path_utils import detect_repo_root
from runtimes_dep_agent.agent.llm_agent import LLMAgent


DEFAULT_CONFIG_PATH = "config-yaml/sample_modelcar_config.yaml"

# Minimal trigger to hand control to the Supervisor.
SUPERVISOR_TRIGGER_MESSAGE = "Start supervisor agent operation. Receive model-car configuration report from config specialist and make deployment decisions."


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the supervisor agent end-to-end."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the model-car YAML config file to preload.",
    )
    parser.add_argument(
        "--service",
        default="Gemini",
        help="Service name to use for the supervisor agent.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Model name to use for the supervisor agent.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    api_key = os.environ.get("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable must be set")
    if args.service == "Self-hosted":
        args.self_hosted_url = os.environ.get("SELF_HOSTED_MODEL_URL")
        if not args.self_hosted_url:
            raise ValueError("SELF_HOSTED_MODEL_URL environment variable must be set for self-hosted service")
    else:
        args.self_hosted_url = None

    # Build the supervisor with preloaded config
    agent = LLMAgent(
        api_key=api_key,
        service=args.service,
        model=args.model,
        base_url=args.self_hosted_url,
        bootstrap_config=args.config,
    )

    print("\nSupervisor Output")
    print("-----------------")

    try:
        result = agent.run_supervisor(SUPERVISOR_TRIGGER_MESSAGE)
        output_text = agent.extract_final_text(result)
    except GraphRecursionError:
        output_text = "Error: maximum recursion depth reached."
    
    summary_path = Path(detect_repo_root(), "info", "supervisor_summary.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(output_text)
    print(output_text)


if __name__ == "__main__":
    main()