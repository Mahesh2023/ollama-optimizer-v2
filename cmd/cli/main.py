"""Ollama Optimizer CLI."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from internal.benchmark.ollama_bench import benchmark_suite
from internal.benchmark.prompts import STANDARD_PROMPTS
from internal.config import get_settings
from internal.hardware.detector import detect_system
from internal.tuner.quant import generate_modelfile, tune_model

app = typer.Typer(help="Ollama Optimizer v2 — LLMOps for local LLM inference.")
console = Console()


@app.command()
def detect() -> None:
    """Detect hardware capabilities."""
    info = detect_system()

    table = Table(title="Hardware Detection", show_lines=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("OS", info.os)
    table.add_row("Arch", info.arch)
    table.add_row("Platform", info.platform)
    table.add_row("CPU", f"{info.cpu_model} ({info.cpu_cores}C/{info.cpu_threads}T)")
    table.add_row("RAM", f"{info.ram_mb:,} MB ({info.ram_mb / 1024:.1f} GB)")
    table.add_row("Apple Silicon", "YES" if info.is_apple_silicon else "NO")
    table.add_row("GPUs", str(len(info.gpus)))
    table.add_row("Total VRAM", f"{info.total_vram_mb:,} MB ({info.total_vram_mb / 1024:.1f} GB)")

    console.print(table)

    if info.gpus:
        gpu_table = Table(title="GPUs", show_lines=False)
        gpu_table.add_column("#", style="cyan")
        gpu_table.add_column("Name")
        gpu_table.add_column("VRAM (MB)")
        gpu_table.add_column("Vendor")
        gpu_table.add_column("Driver")
        for i, g in enumerate(info.gpus):
            gpu_table.add_row(str(i), g.name, f"{g.vram_mb:,}", g.vendor, g.driver_version)
        console.print(gpu_table)


@app.command()
def bench(
    model: str = typer.Option(..., help="Ollama model tag, e.g. llama3.2:3b"),
    base_url: str = typer.Option("http://localhost:11434", help="Ollama base URL"),
    output: Path | None = typer.Option(None, help="Save results as JSON"),
) -> None:
    """Run full benchmark suite against a model."""
    prompts = [(p.name, p.prompt) for p in STANDARD_PROMPTS]

    with console.status(f"Benchmarking {model}..."):
        results = asyncio.run(benchmark_suite(model, prompts, base_url))

    table = Table(title=f"Benchmark Results: {model}")
    table.add_column("Prompt", style="cyan")
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Tokens/sec", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Status")

    for r in results:
        status = "OK" if r.success else f"FAIL: {r.error}"
        table.add_row(
            r.prompt_name,
            f"{r.ttft_ms:.0f}",
            f"{r.tokens_per_sec:.1f}",
            str(r.total_tokens),
            status,
        )
    console.print(table)

    if output:
        output.write_text(json.dumps([r.to_dict() for r in results], indent=2))
        rprint(f"[green]Saved to {output}[/green]")


@app.command()
def tune(
    model: str = typer.Option(..., help="Model name"),
    params_b: float = typer.Option(3.0, help="Model parameters in billions"),
    layers: int = typer.Option(32, help="Number of transformer layers"),
    ctx: int = typer.Option(4096, help="Target context length"),
    modelfile: Path | None = typer.Option(None, help="Write Modelfile here"),
) -> None:
    """Auto-tune model config for current hardware."""
    sys_info = detect_system()
    result = tune_model(model, params_b, sys_info, layers, ctx)

    table = Table(title=f"Tuning Result: {model}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    for key, value in result.to_dict().items():
        if key == "metadata":
            continue
        table.add_row(key, str(value))
    console.print(table)

    if modelfile:
        base = typer.prompt(
            "Base Ollama model", default=f"{model.split(':')[0]}:{result.selected_quant.lower()}"
        )
        modelfile.write_text(generate_modelfile(result, base))
        rprint(f"[green]Wrote Modelfile to {modelfile}[/green]")
        rprint(f"Build with: [cyan]ollama create {model}-opt -f {modelfile}[/cyan]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Enable hot reload"),
) -> None:
    """Start the routing API server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "cmd.server.main:app",
        host=host or settings.server_host,
        port=port or settings.server_port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
