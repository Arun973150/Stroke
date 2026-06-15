#!/usr/bin/env python3
"""
Training Monitor Script
=======================
Real-time monitoring dashboard for nnU-Net training.

This script:
1. Displays live training metrics
2. Shows GPU utilization
3. Estimates time remaining
4. Tracks best checkpoints

Usage:
    python scripts/monitor_training.py --fold 0 --config configs/nnunet_config.yaml
    
    # Auto-refresh every 30 seconds
    python scripts/monitor_training.py --fold 0 --refresh 30
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, setup_nnunet_env, get_dataset_name,
    print_header, print_success, print_warning, print_info, console
)


def get_gpu_stats() -> Dict:
    """Get current GPU statistics."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        name = pynvml.nvmlDeviceGetName(handle)
        
        pynvml.nvmlShutdown()
        
        return {
            'available': True,
            'name': name.decode() if isinstance(name, bytes) else name,
            'memory_used': mem_info.used / (1024**3),
            'memory_total': mem_info.total / (1024**3),
            'memory_percent': mem_info.used / mem_info.total * 100,
            'utilization': util.gpu,
            'temperature': temp
        }
    except Exception:
        pass
    
    # Fallback to torch
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'available': True,
                'name': torch.cuda.get_device_name(0),
                'memory_used': torch.cuda.memory_allocated(0) / (1024**3),
                'memory_total': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'memory_percent': torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100,
                'utilization': None,
                'temperature': None
            }
    except Exception:
        pass
    
    return {'available': False}


def parse_training_log(log_dir: Path) -> List[Dict]:
    """
    Parse nnU-Net training log files.
    
    Args:
        log_dir: Path to fold directory containing logs
        
    Returns:
        List of epoch metrics
    """
    metrics = []
    
    # Find log files
    log_files = sorted(log_dir.glob('training_log_*.txt'))
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    # Parse epoch line (nnU-Net format)
                    if 'epoch:' in line.lower() or 'Epoch' in line:
                        try:
                            # Extract metrics from line
                            parts = line.strip().split()
                            epoch_data = {}
                            
                            for i, part in enumerate(parts):
                                if 'epoch' in part.lower():
                                    # Try to extract epoch number
                                    for p in parts[i:i+3]:
                                        try:
                                            epoch_data['epoch'] = int(p.strip(':,'))
                                            break
                                        except ValueError:
                                            continue
                                elif 'loss' in part.lower():
                                    try:
                                        epoch_data['loss'] = float(parts[i+1].strip(':,'))
                                    except (ValueError, IndexError):
                                        pass
                                elif 'dice' in part.lower():
                                    try:
                                        epoch_data['dice'] = float(parts[i+1].strip(':,'))
                                    except (ValueError, IndexError):
                                        pass
                            
                            if epoch_data:
                                metrics.append(epoch_data)
                        except Exception:
                            pass
        except Exception:
            pass
    
    return metrics


def get_training_status(results_path: Path) -> Dict:
    """
    Get comprehensive training status from nnU-Net results directory.
    
    Args:
        results_path: Path to fold results directory
        
    Returns:
        Status dictionary
    """
    status = {
        'exists': results_path.exists(),
        'current_epoch': None,
        'total_epochs': 1000,
        'best_dice': None,
        'best_epoch': None,
        'train_loss': None,
        'train_dice': None,
        'val_dice': None,
        'start_time': None,
        'last_update': None,
        'checkpoint_final': False,
        'checkpoint_best': False
    }
    
    if not results_path.exists():
        return status
    
    # Check checkpoints
    status['checkpoint_final'] = (results_path / 'checkpoint_final.pth').exists()
    status['checkpoint_best'] = (results_path / 'checkpoint_best.pth').exists()
    
    # Parse debug.json for epoch info
    debug_file = results_path / 'debug.json'
    if debug_file.exists():
        try:
            with open(debug_file, 'r') as f:
                debug = json.load(f)
            status['current_epoch'] = debug.get('current_epoch')
            status.update(debug)
        except Exception:
            pass
    
    # Get last modification time
    if status['checkpoint_best']:
        ckpt_path = results_path / 'checkpoint_best.pth'
        status['last_update'] = datetime.fromtimestamp(ckpt_path.stat().st_mtime)
    
    # Parse training logs for metrics
    metrics = parse_training_log(results_path)
    if metrics:
        latest = metrics[-1]
        status['train_loss'] = latest.get('loss')
        status['train_dice'] = latest.get('dice')
        
        # Find best
        dice_values = [m.get('dice', 0) for m in metrics if m.get('dice')]
        if dice_values:
            status['best_dice'] = max(dice_values)
            status['best_epoch'] = dice_values.index(status['best_dice'])
    
    return status


def display_monitor(config: Dict, fold: int, clear: bool = True) -> None:
    """
    Display training monitor dashboard.
    
    Args:
        config: Configuration dictionary
        fold: Fold number being monitored
        clear: Whether to clear screen before display
    """
    if clear:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    dataset_name = get_dataset_name(config)
    trainer = config['training'].get('trainer', 'nnUNetTrainer')
    plans = config['training'].get('plans', 'nnUNetPlans')
    configuration = config['training']['configuration']
    
    results_path = Path(os.environ['nnUNet_results']) / dataset_name / \
                   f"{trainer}__{plans}__{configuration}" / f"fold_{fold}"
    
    # Get status
    status = get_training_status(results_path)
    gpu_stats = get_gpu_stats()
    
    # Display header
    console.print("╔" + "═" * 68 + "╗")
    console.print("║" + " " * 15 + "[bold cyan]nnU-Net Training Monitor[/bold cyan]" + " " * 25 + "║")
    console.print("╠" + "═" * 68 + "╣")
    
    # Dataset info
    console.print(f"║ Dataset: [bold]{dataset_name}[/bold]" + " " * (50 - len(dataset_name)) + "║")
    console.print(f"║ Config: {configuration} | Fold: {fold}/4" + " " * 35 + "║")
    console.print("╠" + "═" * 68 + "╣")
    
    # Training progress
    if status['exists']:
        current = status.get('current_epoch') or 0
        total = status.get('total_epochs', 1000)
        progress_pct = current / total * 100
        
        # Progress bar
        bar_width = 40
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        console.print(f"║ Epoch: {current}/{total} [{bar}] {progress_pct:.1f}%" + " " * (11 - len(str(current)) - len(str(total))) + "║")
        
        # Metrics
        if status.get('train_dice'):
            console.print(f"║" + " " * 68 + "║")
            console.print(f"║ [bold]Training Metrics:[/bold]" + " " * 50 + "║")
            
            loss_str = f"{status['train_loss']:.4f}" if status.get('train_loss') else "N/A"
            dice_str = f"{status['train_dice']:.4f}" if status.get('train_dice') else "N/A"
            
            console.print(f"║   Loss: {loss_str}  |  Dice: {dice_str}" + " " * 35 + "║")
        
        if status.get('best_dice'):
            best_str = f"{status['best_dice']:.4f}"
            best_epoch = status.get('best_epoch', '?')
            console.print(f"║   Best: {best_str} (epoch {best_epoch})" + " " * 35 + "║")
        
        # Checkpoints
        console.print(f"║" + " " * 68 + "║")
        console.print(f"║ [bold]Checkpoints:[/bold]" + " " * 55 + "║")
        
        final_status = "[green]✓[/green]" if status['checkpoint_final'] else "[red]○[/red]"
        best_status = "[green]✓[/green]" if status['checkpoint_best'] else "[red]○[/red]"
        
        console.print(f"║   {final_status} checkpoint_final.pth   {best_status} checkpoint_best.pth" + " " * 20 + "║")
        
        # Time estimate
        if status.get('last_update') and current > 0:
            elapsed = datetime.now() - status['last_update']
            if current < total:
                remaining = total - current
                # Simple estimate: assume constant time per epoch
                time_per_epoch = 30  # seconds (rough estimate)
                eta_seconds = remaining * time_per_epoch
                eta = timedelta(seconds=eta_seconds)
                console.print(f"║" + " " * 68 + "║")
                console.print(f"║ ETA: ~{eta}" + " " * (58 - len(str(eta))) + "║")
    else:
        console.print(f"║ [yellow]Training not started yet[/yellow]" + " " * 43 + "║")
    
    console.print("╠" + "═" * 68 + "╣")
    
    # GPU stats
    console.print(f"║ [bold]GPU Status:[/bold]" + " " * 56 + "║")
    
    if gpu_stats['available']:
        mem_used = gpu_stats.get('memory_used', 0)
        mem_total = gpu_stats.get('memory_total', 1)
        mem_pct = gpu_stats.get('memory_percent', 0)
        
        console.print(f"║   {gpu_stats['name'][:45]}" + " " * (55 - min(45, len(gpu_stats['name']))) + "║")
        console.print(f"║   Memory: {mem_used:.1f} / {mem_total:.1f} GB ({mem_pct:.1f}%)" + " " * 35 + "║")
        
        if gpu_stats.get('utilization') is not None:
            console.print(f"║   Utilization: {gpu_stats['utilization']}%" + " " * 48 + "║")
        
        if gpu_stats.get('temperature') is not None:
            temp = gpu_stats['temperature']
            temp_color = "green" if temp < 70 else "yellow" if temp < 85 else "red"
            console.print(f"║   Temperature: [{temp_color}]{temp}°C[/{temp_color}]" + " " * 48 + "║")
    else:
        console.print(f"║   [red]No GPU detected[/red]" + " " * 51 + "║")
    
    console.print("╠" + "═" * 68 + "╣")
    
    # Timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"║ Last updated: {now}" + " " * (52 - len(now)) + "║")
    console.print("╚" + "═" * 68 + "╝")
    
    console.print("\n[dim]Press Ctrl+C to exit[/dim]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor nnU-Net training progress"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/nnunet_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--fold', '-f',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help='Fold to monitor'
    )
    parser.add_argument(
        '--refresh', '-r',
        type=int,
        default=30,
        help='Refresh interval in seconds (0 for single display)'
    )
    parser.add_argument(
        '--all-folds',
        action='store_true',
        help='Show status of all folds'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up environment variables
    setup_nnunet_env(config)
    
    if args.all_folds:
        # Show status of all folds
        print_header("nnU-Net Training Status - All Folds")
        
        for fold in range(5):
            dataset_name = get_dataset_name(config)
            trainer = config['training'].get('trainer', 'nnUNetTrainer')
            plans = config['training'].get('plans', 'nnUNetPlans')
            configuration = config['training']['configuration']
            
            results_path = Path(os.environ['nnUNet_results']) / dataset_name / \
                           f"{trainer}__{plans}__{configuration}" / f"fold_{fold}"
            
            status = get_training_status(results_path)
            
            if status['checkpoint_final']:
                print_success(f"Fold {fold}: Complete (best dice: {status.get('best_dice', 'N/A')})")
            elif status['exists']:
                epoch = status.get('current_epoch', '?')
                dice = status.get('train_dice', 'N/A')
                print_warning(f"Fold {fold}: In progress (epoch {epoch}, dice: {dice})")
            else:
                print_info(f"Fold {fold}: Not started")
    else:
        # Monitor single fold
        try:
            while True:
                display_monitor(config, args.fold)
                
                if args.refresh <= 0:
                    break
                
                time.sleep(args.refresh)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()
