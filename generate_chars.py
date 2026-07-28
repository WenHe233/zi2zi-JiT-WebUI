"""
Standalone Evaluation Script for JiT (Just image Transformer)

This script generates images from a test npz file using trained model checkpoints.
Supports:
- Loading and using EMA model parameters
- Distributed generation across multiple GPUs
- Single image output (generated only) or pairwise output (source|generated)
- Unicode-based output filenames (U+XXXX.png) when unicode_labels available in npz

Usage:
    # Single GPU
    python generate_chars.py --checkpoint path/to/checkpoint.pth --test_npz test.npz --output_dir ./output

    # Apple Silicon (PyTorch MPS backend)
    python generate_chars.py --device mps --checkpoint path/to/checkpoint.pth --test_npz test.npz --output_dir ./output

    # Multi-GPU (distributed)
    torchrun --nproc_per_node=4 generate_chars.py --checkpoint path/to/checkpoint.pth --test_npz test.npz

    # Pairwise output (source|generated side by side)
    python generate_chars.py --checkpoint path/to/checkpoint.pth --test_npz test.npz --pairwise target_gen

    # Lazy WebUI request (source/reference images are rendered per batch)
    python generate_chars.py --checkpoint path/to/checkpoint.pth --request-manifest request.json
"""
import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
import cv2

import util.misc as misc
from util.lora_utils import inject_lora, _is_lora_state_dict
from data_processing.font_utils import GlyphRendererPool
from zi2zi_webui.inference import _style_image


DEFAULT_STEPS_BY_METHOD = {
    'euler': 20,
    'heun': 50,
    'ab2': 20,
}


def get_args_parser():
    parser = argparse.ArgumentParser('JiT Evaluation', add_help=False)

    # Required paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--test_npz', type=str,
                        help='Path to a legacy test npz file')
    inputs.add_argument('--request-manifest', type=str,
                        help='Path to a WebUI generation request v2 JSON file')
    parser.add_argument('--output_dir', type=str, default='./eval_output',
                        help='Output directory for generated images')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Execution device (auto prefers cuda, then mps, then cpu)')

    # Model
    parser.add_argument('--model', type=str, default=None,
                        help='Model architecture override (default: use checkpoint model)')

    # Generation parameters
    parser.add_argument('--num_images', type=int, default=None,
                        help='Number of images to generate (default: all)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for reproducible generation (default: 42)')
    parser.add_argument('--num_candidates', type=int, default=1,
                        help='Candidates per input glyph (1-8, default: 1)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Per-GPU batch size for generation')
    parser.add_argument('--cfg', type=float, default=None,
                        help='Classifier-free guidance scale (default: from checkpoint, else 4.0)')
    parser.add_argument('--num_sampling_steps', type=int, default=None,
                        help='Number of ODE sampling steps (default: method-specific when overriding sampler, else from checkpoint)')
    parser.add_argument('--sampling_method', type=str, default=None,
                        choices=['euler', 'heun', 'ab2'],
                        help='ODE solver method (default: from checkpoint, else heun)')
    parser.add_argument('--interval_min', type=float, default=None,
                        help='CFG interval minimum (default: from checkpoint, else 0.0)')
    parser.add_argument('--interval_max', type=float, default=None,
                        help='CFG interval maximum (default: from checkpoint, else 1.0)')

    # LoRA (used only when checkpoint contains LoRA keys)
    parser.add_argument('--lora_r', type=int, default=None,
                        help='LoRA rank override (default: from checkpoint args, else 8)')
    parser.add_argument('--lora_alpha', type=int, default=None,
                        help='LoRA alpha override (default: from checkpoint args, else 16)')
    parser.add_argument('--lora_dropout', type=float, default=None,
                        help='LoRA dropout override (default: from checkpoint args, else 0.0)')
    parser.add_argument('--lora_targets', type=str, default=None,
                        help='Comma-separated LoRA target suffixes override (default: from checkpoint args)')

    # Output options
    parser.add_argument('--pairwise', type=str, default=None,
                        choices=['src_gen', 'target_gen'],
                        help='Save pairwise comparison: src_gen (source|generated) or target_gen (target|generated)')
    # Distributed (required by misc.init_distributed_mode)
    parser.add_argument('--dist_on_itp', action='store_true',
                        help='Use OpenMPI distributed mode (for ITP clusters)')
    parser.add_argument('--dist_url', default='env://',
                        help='URL for distributed training setup')

    return parser


def _mps_is_available():
    return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()


def resolve_device(device_name):
    if device_name == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if _mps_is_available():
            return torch.device('mps')
        return torch.device('cpu')

    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but is not available on this machine.')
    if device_name == 'mps' and not _mps_is_available():
        raise RuntimeError('MPS requested but is not available on this machine.')

    return torch.device(device_name)


def _distributed_env_present():
    return (
        'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    ) or 'SLURM_PROCID' in os.environ or 'OMPI_COMM_WORLD_RANK' in os.environ


def _identity_compile(fn=None, *args, **kwargs):
    if fn is None:
        return lambda inner: inner
    return fn


def patch_torch_for_device(device):
    if device.type == 'cuda':
        return

    if device.type == 'mps':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

    target = torch.device(device)

    def tensor_cuda(self, *args, **kwargs):
        return self.to(target)

    def module_cuda(self, *args, **kwargs):
        return self.to(target)

    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.cuda = module_cuda

    if hasattr(torch, 'compile'):
        torch.compile = _identity_compile

    torch.cuda.amp.autocast = lambda *args, **kwargs: nullcontext()


def _parse_codepoint(value):
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.upper().startswith("U+"):
        text = text[2:]
    return int(text, 16)


class NpzInputSource:
    def __init__(self, path, pairwise):
        self.path = str(Path(path).resolve())
        self.data = np.load(path)
        self.font_labels = self.data['font_labels']
        self.char_labels = self.data['char_labels']
        self.style_images = self.data['style_images']
        self.content_images = self.data['content_images']
        self.unicode_labels = (
            self.data['unicode_labels']
            if 'unicode_labels' in self.data
            else None
        )
        self.target_images = None
        if pairwise == 'target_gen':
            if 'target_images' not in self.data:
                raise ValueError(
                    "target_gen pairwise mode requires 'target_images' in npz file"
                )
            self.target_images = self.data['target_images']
        self.count = len(self.font_labels)
        self.request = None

    def sample(self, index):
        return {
            'font_label': int(self.font_labels[index]),
            'char_label': int(self.char_labels[index]),
            'style_image': self.style_images[index],
            'content_image': self.content_images[index],
            'target_image': (
                self.target_images[index]
                if self.target_images is not None
                else None
            ),
            'unicode_label': (
                int(self.unicode_labels[index])
                if self.unicode_labels is not None
                else None
            ),
        }

    def close(self):
        self.data.close()


class ManifestInputSource:
    def __init__(self, path, pairwise):
        if pairwise == 'target_gen':
            raise ValueError(
                "target_gen pairwise mode is unavailable for generation requests"
            )
        self.path = str(Path(path).resolve())
        self.request = json.loads(Path(path).read_text(encoding='utf-8'))
        if (
            self.request.get('schema_version') != 2
            or self.request.get('kind') != 'zi2zi-generation-request'
        ):
            raise ValueError('Unsupported generation request manifest')
        self.samples = list(self.request.get('samples') or [])
        if not self.samples:
            raise ValueError('Generation request does not contain any samples')
        if self.request.get('count') != len(self.samples):
            raise ValueError('Generation request sample count is inconsistent')
        self.renderer = GlyphRendererPool(
            self.request.get('source_fonts') or [],
            int(self.request.get('resolution', 256)),
        )
        self.style_cache = {}
        self.count = len(self.samples)

    def sample(self, index):
        item = self.samples[index]
        codepoint = _parse_codepoint(item['codepoint'])
        expected_source = Path(item['source_font']).resolve()
        actual_source = self.renderer.source_for(codepoint)
        if actual_source != expected_source:
            raise ValueError(
                f"Source font coverage changed for U+{codepoint:04X}: "
                f"expected {expected_source}, got {actual_source}"
            )
        content = self.renderer.render(codepoint)
        if content is None:
            raise ValueError(f"Cannot render source glyph U+{codepoint:04X}")
        reference = str(Path(item['reference']).resolve())
        if reference not in self.style_cache:
            self.style_cache[reference] = _style_image(reference)
        return {
            'font_label': 0,
            'char_label': 0,
            'style_image': self.style_cache[reference],
            'content_image': np.asarray(content, dtype=np.uint8).transpose(2, 0, 1),
            'target_image': None,
            'unicode_label': codepoint,
        }

    def close(self):
        return None


def expanded_sample_indices(start_idx, batch_size, num_images, candidates):
    """Map an expanded/padded batch to base samples without copying image arrays."""
    expanded = []
    for expanded_index in range(start_idx, start_idx + batch_size):
        real_index = min(expanded_index, num_images - 1)
        expanded.append((real_index // candidates, real_index % candidates))
    return expanded


def main(args):
    device = resolve_device(args.device)
    use_cuda_amp = device.type == 'cuda'
    if not 1 <= args.num_candidates <= 8:
        raise ValueError('--num_candidates must be between 1 and 8')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    if device.type == 'cuda':
        misc.init_distributed_mode(args)
    else:
        if args.dist_on_itp or _distributed_env_present():
            raise RuntimeError('Distributed generation currently requires CUDA. Use single-process mode for MPS/CPU inference.')
        args.distributed = False

    patch_torch_for_device(device)

    from denoiser import Denoiser

    world_size = misc.get_world_size()
    local_rank = misc.get_rank()

    print(f"Rank {local_rank}/{world_size}: Initializing on {device}...")

    # ============ Load Checkpoint ============
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Get args from checkpoint and override model only when explicitly specified
    ckpt_args = checkpoint['args']
    if args.model is not None:
        ckpt_args.model = args.model

    # ============ Create Model ============
    model = Denoiser(ckpt_args)

    # Select state dict (prefer EMA when available)
    if 'model_ema1' in checkpoint:
        print("Using EMA parameters from checkpoint")
        state_dict = checkpoint['model_ema1']
    elif 'model' in checkpoint:
        print("EMA not found, using model parameters")
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # If this is a LoRA checkpoint, inject LoRA wrappers before loading weights.
    is_lora = _is_lora_state_dict(state_dict)
    if is_lora:
        lora_r = args.lora_r if args.lora_r is not None else getattr(ckpt_args, 'lora_r', 8)
        lora_alpha = args.lora_alpha if args.lora_alpha is not None else getattr(ckpt_args, 'lora_alpha', 16)
        lora_dropout = args.lora_dropout if args.lora_dropout is not None else getattr(ckpt_args, 'lora_dropout', 0.0)
        targets_str = args.lora_targets if args.lora_targets is not None else getattr(
            ckpt_args, 'lora_targets', 'qkv,proj,w12,w3'
        )
        targets = [t.strip() for t in targets_str.split(',') if t.strip()]
        replaced = inject_lora(model.net, targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        print(
            "Detected LoRA checkpoint. "
            f"Injected LoRA into {replaced} modules (targets={targets}, r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if is_lora and (missing or unexpected):
        raise RuntimeError(
            "LoRA checkpoint load mismatch after injection: "
            f"missing={len(missing)}, unexpected={len(unexpected)}. "
            "Check --model / --lora_* arguments."
        )
    if missing or unexpected:
        print(f"Warning: state_dict load missing={len(missing)}, unexpected={len(unexpected)}")

    model.to(device)
    model.eval()

    # Resolve generation parameters: CLI override > checkpoint args > hardcoded default
    args.sampling_method = args.sampling_method if args.sampling_method is not None else getattr(ckpt_args, 'sampling_method', 'heun')
    default_steps = DEFAULT_STEPS_BY_METHOD[args.sampling_method]
    args.cfg = args.cfg if args.cfg is not None else getattr(ckpt_args, 'cfg', 4.0)
    if args.num_sampling_steps is None:
        if args.sampling_method == getattr(ckpt_args, 'sampling_method', None):
            args.num_sampling_steps = getattr(ckpt_args, 'num_sampling_steps', default_steps)
        else:
            args.num_sampling_steps = default_steps
    args.interval_min = args.interval_min if args.interval_min is not None else getattr(ckpt_args, 'interval_min', 0.0)
    args.interval_max = args.interval_max if args.interval_max is not None else getattr(ckpt_args, 'interval_max', 1.0)

    # Set generation parameters
    model.cfg_scale = args.cfg
    model.steps = args.num_sampling_steps
    model.method = args.sampling_method
    model.cfg_interval = (args.interval_min, args.interval_max)

    # Print generation config
    print("=" * 50)
    print("  Generation Config")
    print("=" * 50)
    print(f"  Model:           {ckpt_args.model}")
    print(f"  Image size:      {ckpt_args.img_size}")
    print(f"  Checkpoint:      {args.checkpoint}")
    print(f"  LoRA:            {is_lora}")
    print(f"  Sampling:        {args.sampling_method}")
    print(f"  Steps:           {args.num_sampling_steps}")
    print(f"  CFG scale:       {args.cfg}")
    print(f"  CFG interval:    [{args.interval_min}, {args.interval_max}]")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Num images:      {args.num_images or 'all'}")
    print(f"  Pairwise:        {args.pairwise or 'off'}")
    print(f"  World size:      {world_size}")
    print(f"  Device:          {device}")
    print("=" * 50)

    # ============ Load Test Data / Lazy Request ============
    if args.request_manifest:
        print(f"Loading generation request from {args.request_manifest}")
        input_source = ManifestInputSource(args.request_manifest, args.pairwise)
        request_seed = int(input_source.request.get('seed', args.seed))
        request_candidates = int(
            input_source.request.get('candidates', args.num_candidates)
        )
        request_checkpoint = Path(
            input_source.request.get('checkpoint', '')
        ).resolve()
        if request_seed != args.seed:
            raise ValueError(
                f"Request seed {request_seed} does not match --seed {args.seed}"
            )
        if request_candidates != args.num_candidates:
            raise ValueError(
                "Request candidate count does not match --num_candidates"
            )
        if request_checkpoint != Path(args.checkpoint).resolve():
            raise ValueError(
                "Request checkpoint does not match --checkpoint"
            )
    else:
        print(f"Loading test data from {args.test_npz}")
        input_source = NpzInputSource(args.test_npz, args.pairwise)
    if input_source.sample(0)['unicode_label'] is not None:
        print("Loaded unicode labels, will use U+XXXX filenames")
    else:
        print("unicode_labels not found, using index-based filenames")

    num_total_samples = input_source.count * args.num_candidates
    if args.num_candidates > 1:
        print(
            f"Mapped {input_source.count} inputs to {num_total_samples} "
            f"samples ({args.num_candidates} candidates each, base seed={args.seed})"
        )
    num_images = args.num_images if args.num_images else num_total_samples
    num_images = min(num_images, num_total_samples)
    if num_images < 1:
        raise ValueError("No generation samples are available")
    batch_size = args.batch_size

    # Pad to ensure even distribution across GPUs (avoid distributed deadlock)
    padded_num_images = ((num_images + batch_size * world_size - 1) // (batch_size * world_size)) * batch_size * world_size

    num_steps = padded_num_images // (batch_size * world_size)

    print(f"Generating {num_images} images (padded to {padded_num_images}) with batch_size={batch_size}, steps={num_steps}")

    # ============ Create Output Directories ============
    base_folder = os.path.join(
        args.output_dir,
        f"{args.sampling_method}-steps{args.num_sampling_steps}-cfg{args.cfg}-"
        f"interval{args.interval_min}-{args.interval_max}-image{num_images}-res{ckpt_args.img_size}"
    )
    gen_folder = os.path.join(base_folder, "generated")
    compare_folder = os.path.join(base_folder, "compare") if args.pairwise else None

    if local_rank == 0:
        os.makedirs(gen_folder, exist_ok=True)
        if compare_folder:
            os.makedirs(compare_folder, exist_ok=True)
        print(f"Saving to: {base_folder}")

    if world_size > 1:
        dist.barrier()

    # ============ Generation Loop ============
    # Check if this rank has any real images to generate
    first_img_for_rank = local_rank * batch_size
    rank_has_work = first_img_for_rank < num_images

    if not rank_has_work:
        print(f"Rank {local_rank}: No images to generate (num_images={num_images}), skipping.")
        if world_size > 1:
            dist.barrier()
        input_source.close()
        return

    for step in range(num_steps):
        start_idx = world_size * batch_size * step + local_rank * batch_size

        # Skip steps where this rank has no real images left
        if start_idx >= num_images:
            print(f"Rank {local_rank}: Step {step + 1}/{num_steps} — no more images, waiting at barrier.")
            if world_size > 1:
                dist.barrier()
            continue

        print(f"Rank {local_rank}: Generation step {step + 1}/{num_steps}")

        # Render/load only the base samples needed by this expanded batch.
        mapped_indices = expanded_sample_indices(
            start_idx,
            batch_size,
            num_images,
            args.num_candidates,
        )
        sample_cache = {}
        batch_samples = []
        for base_index, _candidate_id in mapped_indices:
            if base_index not in sample_cache:
                sample_cache[base_index] = input_source.sample(base_index)
            batch_samples.append(sample_cache[base_index])
        font_labels_array = np.asarray(
            [sample['font_label'] for sample in batch_samples],
            dtype=np.int64,
        )
        char_labels_array = np.asarray(
            [sample['char_label'] for sample in batch_samples],
            dtype=np.int64,
        )
        style_images_array = np.stack(
            [sample['style_image'] for sample in batch_samples]
        )
        content_images_array = np.stack(
            [sample['content_image'] for sample in batch_samples]
        )

        font_labels_batch = torch.from_numpy(font_labels_array).long().to(device)
        char_labels_batch = torch.from_numpy(char_labels_array).long().to(device)

        style_images_batch = torch.from_numpy(
            style_images_array.copy()
        ).float().to(device)
        style_images_batch = style_images_batch / 255.0 * 2.0 - 1.0

        content_images_batch = torch.from_numpy(
            content_images_array.copy()
        ).float().to(device)
        content_images_batch = content_images_batch / 255.0 * 2.0 - 1.0

        labels = (font_labels_batch, char_labels_batch, style_images_batch, content_images_batch)

        # Generate
        with (torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_cuda_amp else nullcontext()):
            generated = model.generate(labels)

        if world_size > 1:
            dist.barrier()

        # Denormalize: [-1, 1] -> [0, 1]
        generated = (generated + 1) / 2
        generated = generated.detach().cpu()

        # Save images (skip padded samples beyond num_images)
        for b_id in range(batch_size):
            img_id = start_idx + b_id
            if img_id >= num_images:
                break

            # Determine filename
            sample = batch_samples[b_id]
            _base_index, candidate_id = mapped_indices[b_id]
            font_id = int(sample['font_label'])
            if sample['unicode_label'] is not None:
                filename = f"{font_id:04d}_U+{int(sample['unicode_label']):04X}"
            else:
                filename = f"{font_id:04d}_{img_id:05d}"
            if args.num_candidates > 1:
                filename += f"_c{candidate_id:02d}_s{args.seed}"

            # Convert to uint8 BGR for OpenCV
            gen_img = np.round(np.clip(generated[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]  # RGB -> BGR

            # Save generated-only image
            cv2.imwrite(os.path.join(gen_folder, f'{filename}.png'), gen_img)

            # Save pairwise comparison if requested
            if args.pairwise == 'src_gen':
                src_img = sample['content_image'].transpose([1, 2, 0])[:, :, ::-1]  # RGB -> BGR
                pair_img = np.concatenate([src_img, gen_img], axis=1)
                cv2.imwrite(os.path.join(compare_folder, f'{filename}.png'), pair_img)
            elif args.pairwise == 'target_gen':
                target_img = sample['target_image'].transpose([1, 2, 0])[:, :, ::-1]  # RGB -> BGR
                pair_img = np.concatenate([target_img, gen_img], axis=1)
                cv2.imwrite(os.path.join(compare_folder, f'{filename}.png'), pair_img)
        if local_rank == 0:
            current = min((step + 1) * batch_size * world_size, num_images)
            print(
                "WEBUI_PROGRESS "
                + json.dumps(
                    {
                        "current": current,
                        "total": num_images,
                        "message": f"Generated {current}/{num_images} images",
                    }
                ),
                flush=True,
            )

    if world_size > 1:
        dist.barrier()

    if local_rank == 0 and input_source.request is not None:
        generated_files = sorted(
            str(path.relative_to(Path(args.output_dir)))
            for path in Path(gen_folder).glob("*.png")
        )
        result_path = Path(args.output_dir) / "generation-result.json"
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "kind": "zi2zi-generation-result",
                    "request_id": input_source.request.get("request_id"),
                    "region": input_source.request.get("region"),
                    "status": "succeeded",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "seed": args.seed,
                    "candidates": args.num_candidates,
                    "generated_count": len(generated_files),
                    "files": generated_files,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    input_source.close()
    print(f"Rank {local_rank}: Done! Generated images saved to {gen_folder}")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
