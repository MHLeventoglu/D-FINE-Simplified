"""
D-FINE Comprehensive Testing Script
====================================
This script tests and saves every metric including:
- COCO metrics (AP, AR)
- Validator metrics (F1, Precision, Recall, IoU, TPs, FPs, FNs)
- Per-class metrics
- Model info (FLOPs, MACs, Parameters)
- TensorRT FPS/Latency (if TensorRT engine available)

Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import argparse
import contextlib
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS
from src.solver.validator import Validator, scale_boxes
from src.data import CocoEvaluator
from src.data.dataset import mscoco_category2label


class TimeProfiler(contextlib.ContextDecorator):
    """Context manager for timing code execution."""
    def __init__(self):
        self.total = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total += time.time() - self.start

    def reset(self):
        self.total = 0


def get_model_info(cfg) -> Dict:
    """Get model FLOPs, MACs, and parameter count."""
    try:
        from calflops import calculate_flops
        
        class ModelForFlops(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model.deploy() if hasattr(model, 'deploy') else model

            def forward(self, images):
                return self.model(images)

        model = ModelForFlops(cfg.model).eval()
        
        flops, macs, _ = calculate_flops(
            model=model, 
            input_shape=(1, 3, 640, 640), 
            output_as_string=False,
            output_precision=4
        )
        params = sum(p.numel() for p in model.parameters())
        
        return {
            "flops": flops,
            "flops_str": f"{flops / 1e9:.2f} GFLOPs",
            "macs": macs,
            "macs_str": f"{macs / 1e9:.2f} GMACs",
            "parameters": params,
            "parameters_str": f"{params / 1e6:.2f} M"
        }
    except ImportError:
        print("Warning: calflops not installed. Skipping FLOPs calculation.")
        return {"flops": None, "macs": None, "parameters": None}
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs: {e}")
        return {"flops": None, "macs": None, "parameters": None}


def benchmark_pytorch(model, device, input_size=(1, 3, 640, 640), 
                      warmup_iters=50, test_iters=200) -> Dict:
    """Benchmark PyTorch model inference speed."""
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(test_iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000
    fps = 1000 / avg_time
    
    return {
        "pytorch_latency_ms": avg_time,
        "pytorch_latency_std_ms": std_time,
        "pytorch_fps": fps,
        "pytorch_test_iters": test_iters
    }


def benchmark_tensorrt(engine_path: str, input_size=(1, 3, 640, 640),
                       warmup_iters=100, test_iters=1000) -> Optional[Dict]:
    """Benchmark TensorRT engine inference speed."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("Warning: TensorRT or PyCUDA not installed. Skipping TRT benchmark.")
        return None
    
    if not os.path.exists(engine_path):
        print(f"Warning: TensorRT engine not found at {engine_path}")
        return None
    
    try:
        # Load engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        # Allocate buffers
        stream = cuda.Stream()
        bindings = []
        inputs = []
        outputs = []
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = list(engine.get_tensor_shape(name))
            
            # Handle dynamic shapes
            if shape[0] == -1:
                shape[0] = 1
                context.set_input_shape(name, shape)
            
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({"host": host_mem, "device": device_mem, "name": name})
            else:
                outputs.append({"host": host_mem, "device": device_mem, "name": name})
        
        # Set tensor addresses
        for inp in inputs:
            context.set_tensor_address(inp["name"], int(inp["device"]))
        for out in outputs:
            context.set_tensor_address(out["name"], int(out["device"]))
        
        # Prepare input data
        np.copyto(inputs[0]["host"], np.random.randn(*input_size).astype(np.float32).ravel())
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
        
        # Warmup
        for _ in range(warmup_iters):
            context.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()
        
        # Benchmark
        times = []
        for _ in range(test_iters):
            start = time.time()
            context.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        fps = 1000 / avg_time
        
        return {
            "tensorrt_latency_ms": avg_time,
            "tensorrt_latency_std_ms": std_time,
            "tensorrt_fps": fps,
            "tensorrt_test_iters": test_iters,
            "tensorrt_engine_path": engine_path
        }
    
    except Exception as e:
        print(f"Warning: TensorRT benchmark failed: {e}")
        return None


@torch.no_grad()
def evaluate_model(model, criterion, postprocessor, dataloader, evaluator, device) -> Dict:
    """Run full model evaluation and return all metrics."""
    model.eval()
    if criterion:
        criterion.eval()
    
    evaluator.cleanup()
    
    gt: List[Dict[str, torch.Tensor]] = []
    preds: List[Dict[str, torch.Tensor]] = []
    
    print("\nRunning evaluation...")
    for i, (samples, targets) in enumerate(dataloader):
        if i % 50 == 0:
            print(f"  Processing batch {i}/{len(dataloader)}")
        
        samples = samples.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)
        
        res = {target["image_id"].item(): output for target, output in zip(targets, results)}
        if evaluator is not None:
            evaluator.update(res)
        
        # Collect for Validator
        for idx, (target, result) in enumerate(zip(targets, results)):
            gt.append({
                "boxes": scale_boxes(
                    target["boxes"],
                    (target["orig_size"][1], target["orig_size"][0]),
                    (samples[idx].shape[-1], samples[idx].shape[-2]),
                ),
                "labels": target["labels"],
            })
            labels = (
                torch.tensor([mscoco_category2label[int(x.item())] for x in result["labels"].flatten()])
                .to(result["labels"].device)
                .reshape(result["labels"].shape)
            ) if postprocessor.remap_mscoco_category else result["labels"]
            preds.append({
                "boxes": result["boxes"], 
                "labels": labels, 
                "scores": result["scores"]
            })
    
    # Compute Validator metrics
    validator = Validator(gt, preds, conf_thresh=0.5, iou_thresh=0.5)
    validator_metrics = validator.compute_metrics(extended=True)
    
    # Compute COCO metrics
    if evaluator is not None:
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
    
    # Collect all COCO stats
    coco_stats = {}
    metric_names = [
        "AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
        "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge"
    ]
    
    if evaluator is not None and "bbox" in evaluator.coco_eval:
        stats = evaluator.coco_eval["bbox"].stats.tolist()
        for i, name in enumerate(metric_names):
            if i < len(stats):
                coco_stats[name] = stats[i]
    
    return {
        "coco_metrics": coco_stats,
        "validator_metrics": {
            "f1": validator_metrics.get("f1", 0),
            "precision": validator_metrics.get("precision", 0),
            "recall": validator_metrics.get("recall", 0),
            "iou": validator_metrics.get("iou", 0),
            "TPs": validator_metrics.get("TPs", 0),
            "FPs": validator_metrics.get("FPs", 0),
            "FNs": validator_metrics.get("FNs", 0),
        },
        "per_class_metrics": validator_metrics.get("extended_metrics", {}),
        "confusion_matrix": validator.conf_matrix.tolist() if validator.conf_matrix is not None else None,
        "class_mapping": validator.class_to_idx if hasattr(validator, 'class_to_idx') else None
    }


def save_results(results: Dict, output_dir: Path, save_wandb: bool = False):
    """Save all results to files and optionally to W&B."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = output_dir / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")
    
    # Save summary as text
    summary_path = output_dir / "test_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("D-FINE Model Test Results\n")
        f.write(f"Timestamp: {results.get('timestamp', 'N/A')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Model Info
        if results.get("model_info"):
            f.write("MODEL INFORMATION\n")
            f.write("-" * 40 + "\n")
            info = results["model_info"]
            f.write(f"  FLOPs: {info.get('flops_str', 'N/A')}\n")
            f.write(f"  MACs: {info.get('macs_str', 'N/A')}\n")
            f.write(f"  Parameters: {info.get('parameters_str', 'N/A')}\n\n")
        
        # Speed Metrics
        f.write("SPEED METRICS\n")
        f.write("-" * 40 + "\n")
        if results.get("pytorch_benchmark"):
            bench = results["pytorch_benchmark"]
            f.write(f"  PyTorch Latency: {bench.get('pytorch_latency_ms', 0):.2f} ± {bench.get('pytorch_latency_std_ms', 0):.2f} ms\n")
            f.write(f"  PyTorch FPS: {bench.get('pytorch_fps', 0):.2f}\n")
        if results.get("tensorrt_benchmark"):
            bench = results["tensorrt_benchmark"]
            f.write(f"  TensorRT Latency: {bench.get('tensorrt_latency_ms', 0):.2f} ± {bench.get('tensorrt_latency_std_ms', 0):.2f} ms\n")
            f.write(f"  TensorRT FPS: {bench.get('tensorrt_fps', 0):.2f}\n")
        f.write("\n")
        
        # COCO Metrics
        if results.get("evaluation", {}).get("coco_metrics"):
            f.write("COCO METRICS\n")
            f.write("-" * 40 + "\n")
            for name, value in results["evaluation"]["coco_metrics"].items():
                f.write(f"  {name}: {value:.4f}\n")
            f.write("\n")
        
        # Validator Metrics
        if results.get("evaluation", {}).get("validator_metrics"):
            f.write("VALIDATOR METRICS\n")
            f.write("-" * 40 + "\n")
            for name, value in results["evaluation"]["validator_metrics"].items():
                if isinstance(value, float):
                    f.write(f"  {name}: {value:.4f}\n")
                else:
                    f.write(f"  {name}: {value}\n")
            f.write("\n")
    
    print(f"Summary saved to: {summary_path}")
    
    # Save to W&B
    if save_wandb:
        try:
            import wandb
            
            wandb.init(
                project=results.get("config", {}).get("project_name", "D-FINE-Test"),
                name=f"test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=results.get("config", {})
            )
            
            # Log all metrics
            wandb_logs = {}
            
            # Model info
            if results.get("model_info"):
                for k, v in results["model_info"].items():
                    if v is not None and not k.endswith("_str"):
                        wandb_logs[f"model/{k}"] = v
            
            # Speed metrics
            if results.get("pytorch_benchmark"):
                for k, v in results["pytorch_benchmark"].items():
                    if isinstance(v, (int, float)):
                        wandb_logs[f"speed/{k}"] = v
            
            if results.get("tensorrt_benchmark"):
                for k, v in results["tensorrt_benchmark"].items():
                    if isinstance(v, (int, float)):
                        wandb_logs[f"speed/{k}"] = v
            
            # COCO metrics
            if results.get("evaluation", {}).get("coco_metrics"):
                for k, v in results["evaluation"]["coco_metrics"].items():
                    wandb_logs[f"coco/{k}"] = v
            
            # Validator metrics
            if results.get("evaluation", {}).get("validator_metrics"):
                for k, v in results["evaluation"]["validator_metrics"].items():
                    wandb_logs[f"validator/{k}"] = v
            
            # Per-class metrics
            if results.get("evaluation", {}).get("per_class_metrics"):
                for k, v in results["evaluation"]["per_class_metrics"].items():
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v):
                        wandb_logs[f"per_class/{k}"] = v
            
            wandb.log(wandb_logs)
            
            # Save results as artifact
            artifact = wandb.Artifact("test_results", type="results")
            artifact.add_file(str(json_path))
            artifact.add_file(str(summary_path))
            wandb.log_artifact(artifact)
            
            wandb.finish()
            print("Results logged to W&B")
            
        except ImportError:
            print("Warning: wandb not installed. Skipping W&B logging.")
        except Exception as e:
            print(f"Warning: Failed to log to W&B: {e}")


def main(args):
    """Main testing function."""
    print("=" * 60)
    print("D-FINE Comprehensive Model Testing")
    print("=" * 60)
    
    # Setup
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    
    # Load config
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    cfg = YAMLConfig(args.config, **update_dict)
    
    # Set device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\nDevice: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("./test_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize results
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config_path": args.config,
        "checkpoint_path": args.resume,
        "device": str(device),
        "config": dict(cfg.yaml_cfg) if hasattr(cfg, 'yaml_cfg') else {}
    }
    
    # Get model info
    print("\n[1/5] Getting model information...")
    results["model_info"] = get_model_info(cfg)
    print(f"  Parameters: {results['model_info'].get('parameters_str', 'N/A')}")
    print(f"  FLOPs: {results['model_info'].get('flops_str', 'N/A')}")
    
    # Load model
    print("\n[2/5] Loading model...")
    model = cfg.model.to(device)
    
    if args.resume:
        print(f"  Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint and checkpoint["ema"]:
            state_dict = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    # PyTorch benchmark
    print("\n[3/5] Running PyTorch speed benchmark...")
    results["pytorch_benchmark"] = benchmark_pytorch(
        model, device,
        warmup_iters=args.warmup_iters,
        test_iters=args.test_iters
    )
    print(f"  Latency: {results['pytorch_benchmark']['pytorch_latency_ms']:.2f} ms")
    print(f"  FPS: {results['pytorch_benchmark']['pytorch_fps']:.2f}")
    
    # TensorRT benchmark
    print("\n[4/5] Running TensorRT speed benchmark...")
    if args.trt_engine:
        results["tensorrt_benchmark"] = benchmark_tensorrt(
            args.trt_engine,
            warmup_iters=args.warmup_iters,
            test_iters=args.test_iters
        )
        if results["tensorrt_benchmark"]:
            print(f"  Latency: {results['tensorrt_benchmark']['tensorrt_latency_ms']:.2f} ms")
            print(f"  FPS: {results['tensorrt_benchmark']['tensorrt_fps']:.2f}")
    else:
        print("  Skipped (no TensorRT engine provided)")
        results["tensorrt_benchmark"] = None
    
    # Full evaluation
    print("\n[5/5] Running full model evaluation...")
    
    # Setup dataloader and evaluator
    val_dataloader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=False)
    evaluator = cfg.evaluator
    postprocessor = cfg.postprocessor.to(device)
    criterion = cfg.criterion.to(device) if hasattr(cfg, 'criterion') and cfg.criterion else None
    
    results["evaluation"] = evaluate_model(
        model, criterion, postprocessor, val_dataloader, evaluator, device
    )
    
    # Print key metrics
    print("\nKey Results:")
    print("-" * 40)
    if results["evaluation"]["coco_metrics"]:
        print(f"  AP@0.5:0.95: {results['evaluation']['coco_metrics'].get('AP50:95', 0):.4f}")
        print(f"  AP@0.5: {results['evaluation']['coco_metrics'].get('AP50', 0):.4f}")
    print(f"  F1 Score: {results['evaluation']['validator_metrics'].get('f1', 0):.4f}")
    print(f"  Precision: {results['evaluation']['validator_metrics'].get('precision', 0):.4f}")
    print(f"  Recall: {results['evaluation']['validator_metrics'].get('recall', 0):.4f}")
    
    # Save results
    save_results(results, output_dir, save_wandb=args.use_wandb)
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)
    
    dist_utils.cleanup()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D-FINE Comprehensive Testing Script")
    
    # Required arguments
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="Path to config YAML file")
    parser.add_argument("-r", "--resume", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Optional arguments
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="Device to use (default: cuda if available)")
    parser.add_argument("-o", "--output-dir", type=str, default="./test_results",
                        help="Output directory for results")
    parser.add_argument("--trt-engine", type=str, default=None,
                        help="Path to TensorRT engine file for speed benchmark")
    
    # Benchmark settings
    parser.add_argument("--warmup-iters", type=int, default=50,
                        help="Number of warmup iterations for benchmark")
    parser.add_argument("--test-iters", type=int, default=200,
                        help="Number of test iterations for benchmark")
    
    # W&B
    parser.add_argument("--use-wandb", action="store_true",
                        help="Log results to Weights & Biases")
    
    # Config updates
    parser.add_argument("-u", "--update", nargs="+",
                        help="Update YAML config values")
    
    # Distributed settings
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-rank", type=int, default=None)
    
    args = parser.parse_args()
    main(args)
