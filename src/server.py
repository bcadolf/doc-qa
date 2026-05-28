#Starts the MLX LLM server and stops it

import subprocess
import time
import httpx
import os
import sys
from dotenv import load_dotenv
from rich.console import Console
from huggingface_hub import snapshot_download



console = Console()

MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
PORT = 8080

_server_process = None

def ensure_model_local():
    """Confirm the model is available locally, if not download it"""
    console.print(f"[blue]Checking if model {MODEL} is avalible locally...[/blue]")
    try:
        snapshot_download(repo_id=MODEL, local_files_only=True)
        console.print("[green]Model is local and ready[/green]")
    except Exception:
        try:
            console.print("[yellow]Model not found locally, downloading...[/yellow]")
            snapshot_download(repo_id=MODEL)
            console.print("[green]Model is downloaded and ready[/green]")
        except Exception as e:
            console.print(f"[red]Error occurred while downloading model: {e}[/red]")
            raise RuntimeError("Failed to download model")

def start_server():
    """Start MLX server as bg subprocess"""
    global _server_process

    load_dotenv()

    ensure_model_local()

    os.environ["HF_HUB_OFFLINE"] = "1"

    console.print("[blue]Starting MLX server...[/blue]")
    _server_process = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server", "--model", MODEL, "--port", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    _wait_for_server()

def _wait_for_server(timeout: int = 60):
    """wait for MLX server to be ready"""
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = httpx.get(f"http://localhost:{PORT}/v1/models")
            if response.status_code == 200:
                console.print("[green]MLX server is ready[/green]")
                return
        except httpx.ConnectError:
            time.sleep(1)
    raise RuntimeError("MLX server failed to start within timeout")

def stop_server():
    """Stop the MLX server"""
    global _server_process

    if _server_process:
        console.print("[yellow]Shutting down MLX server...[/yellow]")
        _server_process.terminate()
        _server_process.wait()
        _server_process = None
        console.print("[green]MLX server ended[/green]")

