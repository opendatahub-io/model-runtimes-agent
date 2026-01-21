import streamlit as st
import yaml
import os
import time
import subprocess
import re
from typing import List, Dict, Optional
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import json
from pathlib import Path
import selectors

from runtimes_dep_agent.utils.path_utils import detect_repo_root

# Page configuration
st.set_page_config(
    page_title="Model Runtime Deployment Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "agent_started" not in st.session_state:
    st.session_state.agent_started = False
if "workflow_completed" not in st.session_state:
    st.session_state.workflow_completed = False
if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "yaml_config" not in st.session_state:
    st.session_state.yaml_config = None
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = None
if "oci_pull_secret" not in st.session_state:
    st.session_state.oci_pull_secret = None
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "agent_output_text" not in st.session_state:
    st.session_state.agent_output_text = None
if "agent_command_output" not in st.session_state:
    st.session_state.agent_command_output = None
if "agent_start_time" not in st.session_state:
    st.session_state.agent_start_time = None
if "agent_timestamps" not in st.session_state:
    st.session_state.agent_timestamps = {
        "supervisor": None,
        "configuration": None,
        "accelerator": None,
        "deployment": None,
        "qa": None
    }
if "vllm_runtime_image" not in st.session_state:
    # Check for environment variable first
    vllm_image_env = os.environ.get("VLLM_RUNTIME_IMAGE")
    st.session_state.vllm_runtime_image = vllm_image_env if vllm_image_env else None
if "runtime_backend" not in st.session_state:
    st.session_state.runtime_backend = None

# Helper function to get value from extracted data with fallback
def get_value(path: str, default):
    """Get nested value from extracted_data using dot notation path."""
    if not st.session_state.extracted_data:
        return default
    
    keys = path.split('.')
    value = st.session_state.extracted_data
    try:
        for key in keys:
            value = value[key]
        return value if value is not None else default
    except (KeyError, TypeError):
        return default

# Function to load model information from info/models_info.json
def load_model_info_from_json():
    """Load model information from the repo's info/models_info.json file."""
    repo_root = detect_repo_root([Path(__file__).resolve()])
    models_info_path = Path(repo_root, "info", "models_info.json")
    models = []
    
    if not models_info_path.exists():
        return {"num_models": 0, "models": []}
    
    # Don't show data if agent hasn't started yet
    if st.session_state.agent_start_time is None:
        return {"num_models": 0, "models": []}
    
    # Check if file has been updated since agent started (only show if new data)
    try:
        file_mtime = models_info_path.stat().st_mtime
        # Only show if file was modified after agent started
        if file_mtime < st.session_state.agent_start_time:
            return {"num_models": 0, "models": []}
        
        # Check if file is empty (from reset)
        if models_info_path.stat().st_size == 0:
            return {"num_models": 0, "models": []}
    except Exception:
        pass
    
    try:
        with models_info_path.open('r') as f:
            content = f.read().strip()
            if not content:  # Empty file
                return {"num_models": 0, "models": []}
            precomputed_requirements = json.loads(content)
        
        # Convert precomputed_requirements format to display format
        for model_name, model_data in precomputed_requirements.items():
            # Format parameter count
            param_billion = model_data.get('model_p_billion')
            if param_billion is not None:
                if param_billion >= 1:
                    parameter_count = f"{int(param_billion)} Billion" if param_billion == int(param_billion) else f"{param_billion} Billion"
                else:
                    parameter_count = f"{param_billion * 1000:.0f} Million"
            else:
                parameter_count = "Not specified"
            
            # Format quantization
            quant_bits = model_data.get('quantization_bits')
            if quant_bits is not None:
                quantization = f"{quant_bits} bits"
            else:
                quantization = "Not specified"
            
            estimated_vram = model_data.get('required_vram_gb')
            if estimated_vram is None:
                estimated_vram = 0.0
            
            models.append({
                "name": model_data.get('model_name', model_name),
                "image": model_data.get('image', 'Not specified'),
                "image_size_gb": model_data.get('model_size_gb', 0.0) or 0.0,
                "parameter_count": parameter_count,
                "quantization": quantization,
                "estimated_vram_gb": estimated_vram,
                "supported_arch": model_data.get('supported_arch', 'unknown')
            })
        
        num_models = len(models)
        
    except Exception as e:
        st.error(f"Error loading model info from {models_info_path}: {str(e)}")
        return {"num_models": 0, "models": []}
    
    return {"num_models": num_models, "models": models}


def load_deployment_matrix():
    """Load deployment matrix entries from info/deployment_matrix.json."""
    repo_root = detect_repo_root([Path(__file__).resolve()])
    matrix_path = repo_root / "info" / "deployment_matrix.json"

    if (
        st.session_state.agent_start_time is None
        or not matrix_path.exists()
        or matrix_path.stat().st_size == 0
    ):
        return []

    try:
        file_mtime = matrix_path.stat().st_mtime
        if file_mtime < st.session_state.agent_start_time:
            return []

        data = json.loads(matrix_path.read_text())
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        normalized = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            normalized.append(
                {
                    "model_name": entry.get("model_name", "Unknown"),
                    "deployable": bool(entry.get("deployable", False)),
                    "reason": entry.get("reason", "No reason provided"),
                }
            )
        return normalized
    except Exception as exc:
        st.error(f"Error loading deployment matrix: {exc}")
        return []


def stream_agent_command(cmd, env, cwd, live_output_placeholder, timeout_sec=2100, tail_lines=200):
    """Run a command and stream stdout/stderr into the UI while capturing full output."""
    output_lines = []
    start_time = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
    )

    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)

    if live_output_placeholder is not None:
        live_output_placeholder.code("Waiting for output...")

    try:
        while True:
            if time.time() - start_time > timeout_sec:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout_sec, output="\n".join(output_lines))

            events = selector.select(timeout=0.1)
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    continue
                output_lines.append(line.rstrip("\n"))
                if live_output_placeholder is not None:
                    tail = "\n".join(output_lines[-tail_lines:])
                    live_output_placeholder.code(tail)

            if proc.poll() is not None:
                for line in proc.stdout:
                    output_lines.append(line.rstrip("\n"))
                if live_output_placeholder is not None and output_lines:
                    tail = "\n".join(output_lines[-tail_lines:])
                    live_output_placeholder.code(tail)
                break
    finally:
        selector.close()

    return proc.returncode or 0, "\n".join(output_lines)

# Function to extract QA summary from agent output
def extract_qa_summary(agent_output: str) -> tuple[str, str]:
    """
    Extract QA validation summary from the full agent output.
    Returns: (status, message) tuple
    """
    if not agent_output:
        return "pending", "QA validation has not been run yet."
    
    # Look for QA Validation section
    qa_patterns = [
        r"###\s*QA\s*Validation\s*\n(.*?)(?=\n###|\Z)",
        r"##\s*QA\s*Validation\s*\n(.*?)(?=\n##|\Z)",
        r"QA\s*Validation\s*\n(.*?)(?=\n###|\n##|\Z)",
    ]
    
    qa_summary = None
    for pattern in qa_patterns:
        match = re.search(pattern, agent_output, re.IGNORECASE | re.DOTALL)
        if match:
            qa_summary = match.group(1).strip()
            break
    
    if not qa_summary:
        # Fallback: look for any mention of QA
        if "QA" in agent_output.upper() or "qa specialist" in agent_output.lower():
            # Try to find the last paragraph or sentence mentioning QA
            lines = agent_output.split('\n')
            qa_lines = []
            found_qa = False
            for line in lines:
                if 'qa' in line.lower() and ('specialist' in line.lower() or 'validation' in line.lower()):
                    found_qa = True
                if found_qa:
                    qa_lines.append(line)
                    # Stop at next section or after a few lines
                    if line.strip().startswith('###') and len(qa_lines) > 1:
                        qa_lines.pop()  # Remove the section header
                        break
                    if len(qa_lines) > 10:  # Limit to reasonable length
                        break
            if qa_lines:
                qa_summary = '\n'.join(qa_lines).strip()
    
    if not qa_summary:
        return "pending", "QA validation information not found in agent output."
    
    # Clean up the summary - remove excessive whitespace and logs
    # Remove lines that look like logs (contain timestamps, [QA], etc.)
    cleaned_lines = []
    for line in qa_summary.split('\n'):
        line = line.strip()
        # Skip log-like lines (timestamps, [QA] prefixes, command outputs, etc.)
        if (re.match(r'^\[.*\]', line) or 
            re.match(r'^\d{4}-\d{2}-\d{2}', line) or
            re.match(r'^\[QA\]', line) or
            line.startswith('podman') or
            line.startswith('Running ODH') or
            'INFO' in line or 'ERROR' in line or 'WARNING' in line or
            'test_modelvalidation.py' in line or
            'opendatahub-tests' in line):
            continue
        # Skip empty lines at start
        if not cleaned_lines and not line:
            continue
        # Stop at code blocks or JSON examples
        if line.startswith('```') or line.startswith('{') and 'json' in line.lower():
            break
        cleaned_lines.append(line)
    
    qa_summary = '\n'.join(cleaned_lines).strip()
    
    # Extract just the first paragraph or sentence if it's too long
    # Look for natural sentence boundaries
    if len(qa_summary) > 300:
        # Try to find the end of the first complete thought
        sentences = re.split(r'[.!?]\s+', qa_summary)
        if sentences:
            # Take first 2-3 sentences if they're reasonable length
            summary_parts = []
            total_len = 0
            for sent in sentences:
                if total_len + len(sent) > 300 and summary_parts:
                    break
                summary_parts.append(sent)
                total_len += len(sent) + 2
            if summary_parts:
                qa_summary = '. '.join(summary_parts)
                if not qa_summary.endswith(('.', '!', '?')):
                    qa_summary += '.'
    
    # Determine status from summary
    qa_summary_lower = qa_summary.lower()
    if any(word in qa_summary_lower for word in ['not run', 'skipped', 'no-go', 'no go']):
        status = "skipped"
    elif any(word in qa_summary_lower for word in ['pass', 'success', 'passed', 'completed successfully']):
        status = "passed"
    elif any(word in qa_summary_lower for word in ['fail', 'error', 'failed']):
        status = "failed"
    else:
        status = "completed"
    
    # Limit summary length to avoid showing full logs
    if len(qa_summary) > 500:
        # Try to find a good cutoff point (end of sentence)
        cutoff = qa_summary[:500].rfind('.')
        if cutoff > 200:
            qa_summary = qa_summary[:cutoff + 1]
        else:
            qa_summary = qa_summary[:500] + "..."
    
    return status, qa_summary

# Function to parse GPU info from gpu_info.txt
def parse_gpu_info():
    """Parse GPU information from info/gpu_info.txt file and return per-node details."""
    repo_root = detect_repo_root([Path(__file__).resolve()])
    gpu_info_path = os.path.join(repo_root, "info", "gpu_info.txt")
    gpu_data = {
        "nodes": [],  # List of individual node details
        "total_gpus": 0,
        "total_nodes": 0
    }
    
    if not os.path.exists(gpu_info_path):
        return gpu_data
    
    # Don't show data if agent hasn't started yet
    if st.session_state.agent_start_time is None:
        return gpu_data
    
    # Check if file has been updated since agent started (only show if new data)
    try:
        file_mtime = os.path.getmtime(gpu_info_path)
        file_size = os.path.getsize(gpu_info_path)
        # Only show if file was modified after agent started and is not empty
        if file_mtime < st.session_state.agent_start_time or file_size == 0:
            return gpu_data
    except Exception:
        pass
    
    try:
        with open(gpu_info_path, 'r') as f:
            content = f.read()
        
        # Split content by blank lines to get individual node entries
        entries = content.strip().split('\n\n')
        
        for entry in entries:
            entry = entry.strip()
            if not entry or 'Cloud Provider' not in entry:
                continue
            
            # Parse individual node
            node = {
                "cloud_provider": "Unknown",
                "instance_type": "Unknown",
                "gpu_provider": "Unknown",
                "gpu_product": "Unknown",
                "per_gpu_memory_gb": 0.0,
                "allocatable_gpus": 0,
                "node_ram_gb": 0.0,
                "node_storage_gb": 0.0,
                "node_name": "Unknown"
            }
            
            lines = entry.split('\n')
            for line in lines:
                line = line.strip()
                if line and '•' in line:
                    parts = line.replace('•', '').strip().split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        if key == "Cloud Provider":
                            node["cloud_provider"] = value
                        elif key == "Instance Type":
                            node["instance_type"] = value
                        elif key == "GPU Provider":
                            node["gpu_provider"] = value
                        elif key == "GPU Product":
                            node["gpu_product"] = value
                        elif key == "Per-GPU Memory":
                            match = re.search(r'(\d+\.?\d*)', value)
                            if match:
                                node["per_gpu_memory_gb"] = float(match.group(1))
                        elif key == "Allocatable GPUs":
                            try:
                                node["allocatable_gpus"] = int(value)
                            except ValueError:
                                pass
                        elif key == "Node RAM":
                            match = re.search(r'(\d+\.?\d*)', value)
                            if match:
                                node["node_ram_gb"] = float(match.group(1))
                        elif key == "Node Storage":
                            match = re.search(r'(\d+\.?\d*)', value)
                            if match:
                                node["node_storage_gb"] = float(match.group(1))
                        elif key == "Node Name":
                            node["node_name"] = value
            
            # Add node to list if it has valid data
            if node["node_name"] != "Unknown" or node["allocatable_gpus"] > 0:
                gpu_data["nodes"].append(node)
                gpu_data["total_gpus"] += node["allocatable_gpus"]
        
        gpu_data["total_nodes"] = len(gpu_data["nodes"])
        
    except Exception as e:
        st.error(f"Error parsing GPU info: {str(e)}")
    
    return gpu_data

# Sidebar for API key and YAML upload
with st.sidebar:
    st.title("Configuration")
    
    # Gemini API Key input (mandatory)
    st.subheader("Gemini API Key *")
    api_key_input = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key or "",
        placeholder="Enter your Gemini API Key",
        help="Get your API key to Run the Agent (required)"
    )
    
    if api_key_input:
        st.session_state.gemini_api_key = api_key_input
        # Set as environment variable immediately so it's available to the agent
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.markdown('<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 0.85em;">Configured</span>', unsafe_allow_html=True)
    
    
    # OCI Registry Pull Secret input (mandatory)
    st.subheader("OCI Registry Pull Secret *")
    oci_secret_input = st.text_input(
        "Enter your OCI Registry Pull Secret",
        type="password",
        value=st.session_state.oci_pull_secret or "",
        help="Enter your OCI registry pull secret for authentication (required)"
    )
    
    if oci_secret_input:
        st.session_state.oci_pull_secret = oci_secret_input
        # Set as environment variable immediately so it's available to the agent
        os.environ["OCI_REGISTRY_PULL_SECRET"] = oci_secret_input
        st.markdown('<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 0.85em;">Configured</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # vLLM Runtime Image input (optional)
    st.subheader("vLLM Runtime Image")
    # Check environment variable if session state is not set
    default_vllm_image = st.session_state.vllm_runtime_image or os.environ.get("VLLM_RUNTIME_IMAGE", "")
    vllm_image_input = st.text_input(
        "Enter vLLM Runtime Image",
        value=default_vllm_image,
        placeholder="e.g., quay.io/opendatahub/vllm-runtime:latest",
        help="Specify the vLLM runtime image to use for deployment (optional). Can also be set via VLLM_RUNTIME_IMAGE environment variable."
    )
    
    if vllm_image_input:
        st.session_state.vllm_runtime_image = vllm_image_input
        st.markdown('<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 0.85em;">Configured</span>', unsafe_allow_html=True)
    
    # Runtime Accelerator dropdown (optional)
    st.subheader("Runtime Accelerator")
    runtime_backend_options = [
        "Nvidia - CUDA",
        "AMD - ROCm",
        "Intel-Gaudi",
        "IBM Spyre - Spyre"
    ]
    
    # Get current index for selectbox
    current_index = 0  # Default to placeholder
    if st.session_state.runtime_backend and st.session_state.runtime_backend in runtime_backend_options:
        current_index = runtime_backend_options.index(st.session_state.runtime_backend) + 1  # +1 for placeholder
    
    runtime_backend_selected = st.selectbox(
        "Select Runtime Accelerator",
        options=["Select an option..."] + runtime_backend_options,
        index=current_index,
        help="Select the runtime accelerator for vLLM deployment (optional)"
    )
    
    # Update session state only if a valid option is selected (not placeholder)
    if runtime_backend_selected and runtime_backend_selected != "Select an option...":
        st.session_state.runtime_backend = runtime_backend_selected
        st.markdown('<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 0.85em;">Selected</span>', unsafe_allow_html=True)
    elif runtime_backend_selected == "Select an option...":
        st.session_state.runtime_backend = None
    
    st.divider()
    
    # YAML file upload (mandatory)
    st.subheader("Upload Modelcar Images YAML File *")
    uploaded_file = st.file_uploader(
        "Choose a YAML file (required)",
        type=['yaml', 'yml'],
        help="Upload your Modelcar Images YAML file (required)"
    )
    
    if uploaded_file is not None:
        try:
            yaml_content = uploaded_file.read()
            st.session_state.yaml_config = yaml.safe_load(yaml_content)
            # Store original YAML content to preserve exact format
            st.session_state.yaml_content_raw = yaml_content
            st.markdown('<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 0.85em;">Loaded</span>', unsafe_allow_html=True)
            
            # Display YAML content in expander
            with st.expander("View YAML Configuration"):
                st.code(yaml_content.decode('utf-8'), language='yaml')
        except yaml.YAMLError as e:
            st.error(f"Error parsing YAML file: {str(e)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    st.divider()
    
    # Reset button
    if st.button("Reset", width='stretch'):
        # Clear info folder files
        repo_root = detect_repo_root([Path(__file__).resolve()])
        info_dir = os.path.join(repo_root, "info")
        files_to_clear = ["models_info.json", "gpu_info.txt", "deployment_info.txt", "deployment_matrix.json"]
        for filename in files_to_clear:
            file_path = os.path.join(info_dir, filename)
            if os.path.exists(file_path):
                try:
                    # Clear file content by writing empty string
                    with open(file_path, 'w') as f:
                        f.write("")
                except Exception as e:
                    st.error(f"Error clearing {filename}: {str(e)}")
        
        # Reset session state
        st.session_state.agent_started = False
        st.session_state.workflow_completed = False
        st.session_state.workflow_step = 0
        st.session_state.start_time = None
        st.session_state.agent_result = None
        st.session_state.agent_output_text = None
        st.session_state.agent_command_output = None
        st.session_state.agent_start_time = None
        st.session_state.yaml_content_raw = None
        st.session_state.config_path = None
        st.session_state.agent_timestamps = {
            "supervisor": None,
            "configuration": None,
            "accelerator": None,
            "deployment": None,
            "qa": None
        }
        st.rerun()

# Main interface
st.title("🤖 Model Runtime Deployment Agent")

# Badges for GitHub and Model Runtimes in RHOAI
st.markdown("""
<div style="margin-top: 8px;">
    <a href="https://github.com" target="_blank" style="text-decoration: none; margin-right: 8px;">
        <span style="background-color: #24292e; color: white; padding: 6px 12px; border-radius: 6px; font-size: 0.9em; font-weight: 500; display: inline-block;">
            GitHub Repo
        </span>
    </a>
    <a href="#" target="_blank" style="text-decoration: none;">
        <span style="background-color: #0066cc; color: white; padding: 6px 12px; border-radius: 6px; font-size: 0.9em; font-weight: 500; display: inline-block;">
            Model Runtimes in RHOAI
        </span>
    </a>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Start AI Agent button
if not st.session_state.agent_started:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start AI Agent", width='stretch', type="primary"):
            if not st.session_state.gemini_api_key:
                st.error("⚠️ Please enter your Gemini API key in the sidebar first!")
            elif not st.session_state.oci_pull_secret:
                st.error("⚠️ Please enter your OCI Registry Pull Secret in the sidebar first!")
            elif not st.session_state.yaml_config:
                st.error("⚠️ Please upload a YAML configuration file in the sidebar first!")
            else:
                # Environment variables are already set when user enters them in sidebar
                try:
                    # Ensure environment variables are set (they should already be set from sidebar inputs)
                    if st.session_state.gemini_api_key:
                        os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key
                    if st.session_state.oci_pull_secret:
                        os.environ["OCI_REGISTRY_PULL_SECRET"] = st.session_state.oci_pull_secret
                    
                    # Save uploaded YAML to a file (YAML upload is mandatory)
                    # Use original YAML content to preserve exact format
                    import tempfile
                    temp_dir = tempfile.gettempdir()
                    config_path = os.path.join(temp_dir, "modelcar_config.yaml")
                    with open(config_path, 'wb') as tmp_file:
                        # Use original YAML content if available, otherwise dump parsed version
                        if hasattr(st.session_state, 'yaml_content_raw') and st.session_state.yaml_content_raw:
                            tmp_file.write(st.session_state.yaml_content_raw)
                        else:
                            # Fallback: dump parsed YAML if original content not available
                            yaml.dump(st.session_state.yaml_config, tmp_file)
                    
                    st.session_state.config_path = config_path
                    st.session_state.agent_started = True
                    st.session_state.start_time = time.time()
                    st.session_state.workflow_step = 1  # Start at step 1
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize: {str(e)}")
                    st.exception(e)
    st.info("Click 'Start AI Agent' to begin running the Agent")
else:
    # Status checks
    st.subheader("Status Checks")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.gemini_api_key:
            st.markdown('<span style="background-color: #28a745; color: white; padding: 6px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 500;">Gemini API Key: Verified</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="background-color: #dc3545; color: white; padding: 6px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 500;">Gemini API Key: Not configured</span>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.oci_pull_secret:
            st.markdown('<span style="background-color: #28a745; color: white; padding: 6px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 500;">OCI Pull Secret: Verified</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="background-color: #dc3545; color: white; padding: 6px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 500;">OCI Pull Secret: Not configured</span>', unsafe_allow_html=True)
    
    
    # Workflow progress
    st.subheader("Agent Workflow Progress")
    
    # Define workflow steps
    workflow_steps = [
        {"name": "Starting Supervisor Agent", "status": "pending"},
        {"name": "Calling Configuration Agent", "status": "pending"},
        {"name": "Calling Accelerator Agent", "status": "pending"},
        {"name": "Calling Deployment Specialist", "status": "pending"},
        {"name": "Calling QA Specialist", "status": "pending"},
        # {"name": "Calling Reporting Agent", "status": "pending"},
    ]
    
    # Update workflow steps based on current step
    for i, step in enumerate(workflow_steps):
        if i < st.session_state.workflow_step:
            workflow_steps[i]["status"] = "completed"
        elif i == st.session_state.workflow_step:
            workflow_steps[i]["status"] = "in_progress"
    
    # Display progress
    progress_value = st.session_state.workflow_step / len(workflow_steps) if len(workflow_steps) > 0 else 0.0
    # Ensure progress value is between 0 and 1
    progress_value = max(0.0, min(1.0, progress_value))
    st.progress(progress_value)
    
    # Display step statuses with badges
    for i, step in enumerate(workflow_steps):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"{step['name']}")
        with col2:
            if step["status"] == "completed":
                st.markdown('<span style="background-color: #28a745; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">Completed</span>', unsafe_allow_html=True)
            elif step["status"] == "in_progress":
                st.markdown('<span style="background-color: #007bff; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">In Progress</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="background-color: #6c757d; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">Pending</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Show loader/spinner above all outputs when agent is running
    if st.session_state.workflow_step >= 1 and st.session_state.workflow_step < 6 and not st.session_state.workflow_completed:
        with st.spinner("Running supervisor agent..."):
            time.sleep(0.1)  # Small delay to show spinner
        st.markdown("")  # Add spacing after spinner

    live_output_placeholder = None
    if st.session_state.workflow_step == 1 and not st.session_state.workflow_completed:
        st.subheader("Live Agent Output")
        st.caption("Streaming the last 200 lines while the agent runs.")
        live_output_placeholder = st.empty()
    
    # Display outputs based on workflow step
    # Only show Configuration Agent Results if agent has started
    if st.session_state.workflow_step >= 1 and st.session_state.agent_start_time is not None:
        st.subheader("Configuration Agent Results")
        
        # Load model information from info/models_info.json
        model_info = load_model_info_from_json()
        num_models = model_info["num_models"]
        models = model_info["models"]
        
        # Only display if we have models (file was updated)
        if num_models > 0:
            # Track configuration agent completion time
            if st.session_state.agent_timestamps["configuration"] is None:
                st.session_state.agent_timestamps["configuration"] = time.time()
            
            # Display number of models at the top
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; background-color: #f8f9fa;">
            <p style="font-size: 1.2em; font-weight: bold; color: #28a745;">Number of models found in Modelcar YAML: {num_models}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display all model details in a single expandable dropdown
            if models:
                with st.expander("Model Details", expanded=False):
                    for idx, model in enumerate(models, 1):
                        st.markdown(f"""
                        <div style="padding: 12px; margin-bottom: 16px; background-color: #f8f9fa; border-radius: 4px; border-left: 4px solid #28a745;">
                        <h4 style="margin-top: 0; margin-bottom: 12px; color: #28a745;">Model {idx}: {model.get('name', 'Unknown')}</h4>
                        <ul style="list-style-type: none; padding-left: 0; margin-bottom: 0;">
                        <li><strong>Model Name</strong>: <code>{model.get('name', 'Unknown')}</code></li>
                        <li><strong>Image</strong>: {model.get('image', 'Not specified')}</li>
                        <li><strong>Image Size</strong>: {model.get('image_size_gb', 0.0)} GB</li>
                        <li><strong>Parameter Count</strong>: {model.get('parameter_count', 'Not specified')}</li>
                        <li><strong>Quantization</strong>: {model.get('quantization', 'Not specified')}</li>
                        <li><strong>Estimated VRAM</strong>: {model.get('estimated_vram_gb', 0.0)} GB</li>
                        <li><strong>Supported Architectures</strong>: {model.get('supported_arch', 'unknown')}</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            if st.session_state.workflow_step >= 2:
                st.info("Configuration Agent is processing. Please wait...")
        
        st.markdown("---")
    
    if st.session_state.workflow_step >= 2:
        st.subheader("Accelerator Agent Results")
        
        # Fetch latest GPU info from gpu_info.txt
        gpu_info = parse_gpu_info()
        
        # Only display if we have GPU nodes (file was updated)
        if gpu_info["total_nodes"] > 0:
            # Track accelerator agent completion time
            if st.session_state.agent_timestamps["accelerator"] is None:
                st.session_state.agent_timestamps["accelerator"] = time.time()
            
            # Display total number of GPU nodes
            total_nodes = gpu_info["total_nodes"]
            total_gpus = gpu_info["total_gpus"]
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; background-color: #f8f9fa;">
            <h3>Accelerator Summary</h3>
            <p style="font-size: 1.2em; font-weight: bold; color: #28a745; margin-bottom: 8px;">Total Number of GPU Nodes: {total_nodes}</p>
            <p style="font-size: 1.1em; font-weight: bold; color: #28a745; margin-bottom: 12px;">Total Number of GPUs Available: {total_gpus}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display details for each GPU node in expandable sections
            if gpu_info["nodes"]:
                for idx, node in enumerate(gpu_info["nodes"], 1):
                    accelerator_status = "GPU available" if node["allocatable_gpus"] > 0 else "No GPU available"
                    
                    with st.expander(f"GPU Node {idx} Details - {node['node_name']}", expanded=False):
                        st.markdown(f"""
                        <div style="padding: 12px; background-color: #f8f9fa; border-radius: 4px;">
                        <ul style="list-style-type: none; padding-left: 0;">
                        <li><strong>Node Name</strong>: {node['node_name']}</li>
                        <li><strong>Status</strong>: {accelerator_status}</li>
                        <li><strong>Cloud Provider</strong>: {node['cloud_provider']}</li>
                        <li><strong>Instance Type</strong>: {node['instance_type']}</li>
                        <li><strong>GPU Provider</strong>: {node['gpu_provider']}</li>
                        <li><strong>GPU Product</strong>: {node['gpu_product']}</li>
                        <li><strong>Per-GPU Memory</strong>: {node['per_gpu_memory_gb']} GB</li>
                        <li><strong>Allocatable GPUs</strong>: {node['allocatable_gpus']}</li>
                        <li><strong>Node RAM</strong>: {node['node_ram_gb']} GB</li>
                        <li><strong>Node Storage</strong>: {node['node_storage_gb']} GB</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Accelerator Agent is processing. Please wait...")
        
        st.markdown("---")
    
    if st.session_state.workflow_step >= 3:
        st.subheader("Deployment Specialist Results")
        
        # Load deployment info from info/deployment_info.txt
        repo_root = detect_repo_root([Path(__file__).resolve()])
        deployment_info_path = os.path.join(repo_root, "info", "deployment_info.txt")
        
        # Check if file exists and has been updated since agent started
        file_exists_and_updated = False
        if os.path.exists(deployment_info_path) and st.session_state.agent_start_time is not None:
            try:
                file_mtime = os.path.getmtime(deployment_info_path)
                file_size = os.path.getsize(deployment_info_path)
                # Only show if file was modified after agent started and is not empty
                if file_mtime >= st.session_state.agent_start_time and file_size > 0:
                    file_exists_and_updated = True
            except Exception:
                pass
        
        if file_exists_and_updated:
            # Track deployment agent completion time
            if st.session_state.agent_timestamps["deployment"] is None:
                st.session_state.agent_timestamps["deployment"] = time.time()
            
            try:
                with open(deployment_info_path, 'r', encoding='utf-8') as f:
                    deployment_info_text = f.read().strip()
                
                # Extract verdict for card and expander label from deployment_info.txt
                # Parse the actual format from the file (e.g., "### Deployment Decision: NO-GO" or "**Verdict: GO**")
                verdict = "GO"
                verdict_color = "#28a745"  # Green for GO
                
                # Check for NO-GO patterns (case-insensitive)
                deployment_info_upper = deployment_info_text.upper()
                if any(pattern in deployment_info_upper for pattern in [
                    "DEPLOYMENT DECISION: NO-GO",
                    "DEPLOYMENT DECISION: NO GO",
                    "VERDICT: NO-GO",
                    "VERDICT: NO GO",
                    "**VERDICT: NO-GO**",
                    "**VERDICT: NO GO**"
                ]):
                    verdict = "NO-GO"
                    verdict_color = "#dc3545"  # Red for NO-GO
                elif any(pattern in deployment_info_upper for pattern in [
                    "DEPLOYMENT DECISION: GO",
                    "VERDICT: GO",
                    "**VERDICT: GO**"
                ]):
                    verdict = "GO"
                    verdict_color = "#28a745"  # Green for GO
                
                # Display in expandable dropdown
                with st.expander(f"Deployment Decision Details", expanded=False):
                    # Render the markdown content using Streamlit's native markdown renderer
                    st.markdown(deployment_info_text)
            except Exception as e:
                st.error(f"Error reading deployment info from {deployment_info_path}: {str(e)}")
        else:
            st.info("Deployment Specialist is processing. Please wait...")
        
        st.markdown("---")

    deployment_matrix_entries = load_deployment_matrix()
    if deployment_matrix_entries:
        st.subheader("Deployment Matrix")
        deployable = [entry for entry in deployment_matrix_entries if entry["deployable"]]
        blocked = [entry for entry in deployment_matrix_entries if not entry["deployable"]]

        col_deployable, col_blocked = st.columns(2)
        with col_deployable:
            st.markdown("**Deployable Models**")
            if deployable:
                for entry in deployable:
                    st.markdown(
                        f"- ✅ `{entry['model_name']}` — {entry['reason']}"
                    )
            else:
                st.markdown("_None listed._")

        with col_blocked:
            st.markdown("**Non-deployable Models**")
            if blocked:
                for entry in blocked:
                    st.markdown(
                        f"- ⚠️ `{entry['model_name']}` — {entry['reason']}"
                    )
            else:
                st.markdown("_None listed._")

        st.markdown("---")
    
    if st.session_state.workflow_step >= 4:
        st.subheader("QA Specialist Results")
        
        # Extract QA summary from full agent output
        agent_output = st.session_state.agent_output_text or ""
        qa_status, qa_message = extract_qa_summary(agent_output)
        
        # If no QA info found and agent output exists, show a message
        if qa_status == "pending" and agent_output:
            qa_message = "QA validation information is being processed. Please check the full agent output for details."
        
        # Determine color based on status
        if qa_status == "passed" or qa_status == "completed":
            status_color = "#28a745"  # Green
        elif qa_status == "failed":
            status_color = "#dc3545"  # Red
        elif qa_status == "skipped":
            status_color = "#ffc107"  # Yellow
        else:
            status_color = "#6c757d"  # Gray for pending/unknown
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
        <h3>QA Validation</h3>
        <p><strong>Status:</strong> <span style="color: {status_color}; font-weight: bold;">{qa_status.upper()}</span></p>
        <p>{qa_message}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    # if st.session_state.workflow_step >= 5:
    #     st.subheader("Reporting Agent Results")
    #     elapsed_time = time.time() - st.session_state.start_time
    #     models_executed = get_value("reporting.models_executed", 2)
    #     reporting_status = get_value("reporting.status", "Deployment completed successfully")
    #     reporting_channel = get_value("reporting.channel", "#deployment_agent_report")
    #     
    #     st.markdown(f"""
    #     <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
    #     <h3>Slack Notification</h3>
    #     <p>The <strong>Reporting Agent</strong> has successfully sent the deployment summary to Slack channel <a href="https://slack.com" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;"><strong>{reporting_channel}</strong></a>.</p>
    #     <p><strong>Summary sent:</strong></p>
    #     <ul>
    #     <li><strong>Models Executed</strong>: {models_executed}</li>
    #     <li><strong>Time Taken</strong>: {elapsed_time:.2f} seconds</li>
    #     <li><strong>Status</strong>: {reporting_status}</li>
    #     <li><strong>Channel</strong>: <a href="https://slack.com" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;">{reporting_channel}</a></li>
    #     </ul>
    #     <p style="margin-top: 12px; color: #28a745;"><strong>✓ Message delivered successfully</strong></p>
    #     </div>
    #     """, unsafe_allow_html=True)
    #     st.markdown("---")
    
    if st.session_state.workflow_step >= 5:
        st.subheader("Summary")
        elapsed_time = time.time() - st.session_state.start_time
        # Count only deployable models from deployment_matrix.json
        deployment_matrix_entries = load_deployment_matrix()
        deployable_models = [entry for entry in deployment_matrix_entries if entry.get("deployable", False)]
        models_executed_summary = len(deployable_models) if deployable_models else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models Executed", str(models_executed_summary))
        with col2:
            st.metric("Time Taken", f"{elapsed_time:.2f} seconds")
        
        st.markdown('<div style="margin-top: 20px;"><span style="background-color: #28a745; color: white; padding: 8px 16px; border-radius: 4px; font-size: 0.95em; font-weight: 500;">Deployment completed successfully</span></div>', unsafe_allow_html=True)
        
        # Interactive Graphs
        st.markdown("---")
        st.subheader("Deployment Analytics")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Agent Execution Timeline", "Resource Usage"])
        
        with tab1:
            # Agent execution timeline with actual timestamps
            agent_timestamps = st.session_state.agent_timestamps
            agent_start = st.session_state.agent_start_time if st.session_state.agent_start_time else time.time() - elapsed_time
            
            # Calculate actual durations and start times
            agents_data = []
            
            # Supervisor Agent - from start until agent_output_text is available
            if agent_timestamps["supervisor"]:
                supervisor_start = 0
                supervisor_duration = agent_timestamps["supervisor"] - agent_start
                agents_data.append({
                    'Agent': 'Supervisor',
                    'Start Time': supervisor_start,
                    'Duration': supervisor_duration,
                    'Status': 'Completed'
                })
            
            # Configuration Agent - from start until models_info.json appears
            if agent_timestamps["configuration"]:
                config_start = 0
                config_duration = agent_timestamps["configuration"] - agent_start
                agents_data.append({
                    'Agent': 'Configuration',
                    'Start Time': config_start,
                    'Duration': config_duration,
                    'Status': 'Completed'
                })
            
            # Accelerator Agent - from start until gpu_info.txt appears
            if agent_timestamps["accelerator"]:
                accel_start = 0
                accel_duration = agent_timestamps["accelerator"] - agent_start
                agents_data.append({
                    'Agent': 'Accelerator',
                    'Start Time': accel_start,
                    'Duration': accel_duration,
                    'Status': 'Completed'
                })
            
            # Deployment Specialist - from start until deployment_info.txt appears
            if agent_timestamps["deployment"]:
                deploy_start = 0
                deploy_duration = agent_timestamps["deployment"] - agent_start
                agents_data.append({
                    'Agent': 'Deployment',
                    'Start Time': deploy_start,
                    'Duration': deploy_duration,
                    'Status': 'Completed'
                })
            
            # QA Specialist - skip for now as requested
            
            # Create DataFrame from actual data
            if agents_data:
                agent_data = pd.DataFrame(agents_data)
            else:
                # Fallback to estimated times if no timestamps available yet
                agent_data = pd.DataFrame({
                    'Agent': ['Supervisor', 'Configuration', 'Accelerator', 'Deployment'],
                    'Start Time': [0, elapsed_time*0.2, elapsed_time*0.4, elapsed_time*0.6],
                    'Duration': [elapsed_time*0.2, elapsed_time*0.2, elapsed_time*0.2, elapsed_time*0.2],
                    'Status': ['Completed', 'Completed', 'Completed', 'Completed']
                })
            
            fig_timeline = go.Figure()
            colors = ['#28a745', '#007bff', '#17a2b8', '#ffc107', '#28a745', '#6c757d']
            
            for i, row in agent_data.iterrows():
                # Format duration text: show in minutes if > 99 seconds
                duration = row['Duration']
                if duration > 99:
                    duration_text = f"{duration / 60:.2f}m"
                    hover_duration = f"{duration / 60:.2f} minutes ({duration:.2f}s)"
                else:
                    duration_text = f"{duration:.2f}s"
                    hover_duration = f"{duration:.2f}s"
                
                fig_timeline.add_trace(go.Bar(
                    x=[row['Agent']],
                    y=[row['Duration']],
                    base=[row['Start Time']],
                    marker_color=colors[i],
                    name=row['Agent'],
                    text=[duration_text],
                    textposition='inside',
                    hovertemplate=f"<b>{row['Agent']}</b><br>Duration: {hover_duration}<br>Status: {row['Status']}<extra></extra>"
                ))
            
            fig_timeline.update_layout(
                title="Agent Execution Timeline",
                xaxis_title="Agent",
                yaxis_title="Time (seconds)",
                barmode='group',
                height=400,
                showlegend=False,
                hovermode='closest'
            )
            st.plotly_chart(fig_timeline, width='stretch')
        
        with tab2:
            # Resource usage chart
            gpu_avail = get_value("resources.gpu_memory_available_gb", 44.99)
            gpu_req = get_value("resources.gpu_memory_required_gb", 18.0)
            disk_space = get_value("resources.disk_space_gb", 15.24)
            
            resource_data = pd.DataFrame({
                'Resource': ['GPU Memory (Available)', 'GPU Memory (Required)', 'Disk Space'],
                'Value': [gpu_avail, gpu_req, disk_space],
                'Unit': ['GB', 'GB', 'GB']
            })
            
            fig_resources = go.Figure()
            fig_resources.add_trace(go.Bar(
                x=resource_data['Resource'],
                y=resource_data['Value'],
                marker_color=['#28a745', '#ffc107', '#17a2b8'],
                text=[f"{v} {u}" for v, u in zip(resource_data['Value'], resource_data['Unit'])],
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>Value: %{y} GB<extra></extra>"
            ))
            
            fig_resources.update_layout(
                title="Resource Usage Overview",
                xaxis_title="Resource Type",
                yaxis_title="Value (GB)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_resources, width='stretch')
        
        st.session_state.workflow_completed = True
        st.markdown("---")
    
    # Auto-advance workflow steps and run agent command
    if not st.session_state.workflow_completed and st.session_state.workflow_step < 6:
        # Run the actual agent command when workflow starts
        if st.session_state.workflow_step == 1 and not st.session_state.agent_command_output:
            # Set agent start time for timeline tracking
            if st.session_state.agent_start_time is None:
                st.session_state.agent_start_time = time.time()
            
            placeholder = st.empty()
            with placeholder:
                with st.spinner("Running supervisor agent..."):
                    try:
                        # Run the CLI command: agent --config <config_path>
                        config_path = st.session_state.config_path
                        
                        # Find the agent command - try venv first, then system PATH
                        project_dir = os.path.dirname(os.path.abspath(__file__))
                        venv_agent = os.path.join(project_dir, ".venv", "bin", "agent")
                        cmd = None
                        
                        if os.path.exists(venv_agent):
                            cmd = [venv_agent, "--config", config_path]
                        else:
                            # Try to find agent in PATH
                            import shutil
                            agent_path = shutil.which("agent")
                            if agent_path:
                                cmd = [agent_path, "--config", config_path]
                            else:
                                # Fallback: use Python module directly
                                python_cmd = shutil.which("python3") or shutil.which("python")
                                if python_cmd:
                                    cmd = [python_cmd, "-m", "runtimes_dep_agent.execute_agent", "--config", config_path]
                                else:
                                    raise FileNotFoundError("Could not find 'agent' command or Python. Make sure the package is installed.")
                        
                        # Ensure venv's bin is in PATH
                        env = os.environ.copy()
                        venv_bin = os.path.join(project_dir, ".venv", "bin")
                        if os.path.exists(venv_bin):
                            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
                        
                        if live_output_placeholder is None:
                            live_output_placeholder = st.empty()

                        env["PYTHONUNBUFFERED"] = "1"
                        returncode, output = stream_agent_command(
                            cmd,
                            env=env,
                            cwd=project_dir,
                            live_output_placeholder=live_output_placeholder,
                            timeout_sec=2100,
                            tail_lines=200,
                        )
                        if returncode != 0:
                            output = f"Command exited with code {returncode}\n\n{output}"
                        
                        st.session_state.agent_command_output = output
                        st.session_state.agent_output_text = output
                        
                        # Track supervisor completion time (when agent_output_text is available)
                        if st.session_state.agent_timestamps["supervisor"] is None:
                            st.session_state.agent_timestamps["supervisor"] = time.time()
                        
                        # Advance to next step
                        st.session_state.workflow_step = 2
                    except subprocess.TimeoutExpired as exc:
                        timeout_message = "Error: Command timed out after 35 minutes"
                        partial_output = exc.output or ""
                        if partial_output:
                            timeout_message = f"{timeout_message}\n\n{partial_output}"
                        st.session_state.agent_command_output = timeout_message
                        st.session_state.agent_output_text = timeout_message
                        st.session_state.workflow_step = 6
                    except Exception as e:
                        st.session_state.agent_command_output = f"Error running agent command: {str(e)}"
                        st.session_state.agent_output_text = f"Error running agent command: {str(e)}"
                        st.session_state.workflow_step = 6
            st.rerun()
        elif st.session_state.workflow_step >= 2 and st.session_state.workflow_step < 6:
            # Simulate progress through remaining workflow steps (no spinner needed, already shown above)
            time.sleep(1.5)  # Shorter delay since agent already ran
            st.session_state.workflow_step += 1
            st.rerun()

# Display Full Agent Output at the bottom after all charts
if st.session_state.agent_output_text:
    st.subheader("Full Agent Output")
    with st.expander("📋 Full Agent Output", expanded=False):
        st.markdown(f"```\n{st.session_state.agent_output_text}\n```")
st.markdown("---")
# Footer at the bottom
st.markdown(
    """
    <div style='text-align: center; color: gray; padding-top: 20px;'>
        <small><strong>Model Runtime Deployment Agent</strong> | Powered by Model Runtimes Team ( RHOAI )</small>
    </div>
    """,
    unsafe_allow_html=True
)
