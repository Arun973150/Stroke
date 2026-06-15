"""
Logging Utilities
=================
Consistent logging setup for Sentinel Stroke pipeline.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


# Rich console for pretty output
console = Console()


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        log_to_file: Whether to also log to file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with Rich formatting
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        rich_tracebacks=True
    )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def print_header(title: str, width: int = 70) -> None:
    """Print a formatted header."""
    console.print("=" * width)
    console.print(f"[bold cyan]{title}[/bold cyan]")
    console.print("=" * width)


def print_step(arg1: any, arg2: Optional[int] = None, arg3: Optional[str] = None) -> None:
    """
    Print a step indicator.
    
    Supports:
    - print_step(step_num, total_steps, description)
    - print_step(description)
    """
    if isinstance(arg1, int) and arg2 is not None and arg3 is not None:
        console.print(f"\n[bold green][{arg1}/{arg2}][/bold green] {arg3}")
    else:
        # Fallback to description-only mode or info style
        message = str(arg1)
        console.print(f"\n[bold green]>[/bold green] {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green][OK][/bold green] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow][WARN][/bold yellow] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red][ERR][/bold red] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue][INFO][/bold blue] {message}")


def get_progress_bar() -> Progress:
    """Get a rich progress bar instance."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    )


class TrainingLogger:
    """Logger specifically for training progress."""
    
    def __init__(self, log_dir: str, experiment_name: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            experiment_name, 
            str(self.log_dir), 
            level=logging.INFO
        )
        
        # Metrics history
        self.metrics_history = []
        
    def log_epoch(
        self, 
        epoch: int, 
        train_loss: float, 
        train_dice: float,
        val_loss: Optional[float] = None,
        val_dice: Optional[float] = None,
        lr: Optional[float] = None
    ) -> None:
        """Log epoch metrics."""
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_dice': train_dice,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'lr': lr,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics_history.append(metrics)
        
        # Console output
        msg = f"Epoch {epoch:4d} | train_loss: {train_loss:.4f} | train_dice: {train_dice:.4f}"
        if val_loss is not None:
            msg += f" | val_loss: {val_loss:.4f} | val_dice: {val_dice:.4f}"
        if lr is not None:
            msg += f" | lr: {lr:.6f}"
        
        self.logger.info(msg)
        
    def log_checkpoint(self, epoch: int, metric: float, path: str) -> None:
        """Log checkpoint save."""
        self.logger.info(f"Checkpoint saved at epoch {epoch} (metric: {metric:.4f}): {path}")
        
    def log_best_model(self, epoch: int, metric: float) -> None:
        """Log new best model."""
        console.print(f"[bold green][BEST][/bold green] New best model at epoch {epoch} with dice: {metric:.4f}")
        self.logger.info(f"New best model at epoch {epoch} with dice: {metric:.4f}")


if __name__ == "__main__":
    # Test logging
    print_header("Logging Test")
    print_step(1, 3, "Testing logging utilities")
    print_success("Success message")
    print_warning("Warning message")
    print_error("Error message")
    print_info("Info message")
