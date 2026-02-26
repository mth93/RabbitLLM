from __future__ import annotations

import json
import logging
import os
import time
import gc
import shutil
from glob import glob
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import huggingface_hub
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from ..persist import ModelPersister
from .compression import (
    bitsandbytes_installed,
    compress_layer_state_dict,
    uncompress_layer_state_dict,
)
from .memory import NotEnoughSpaceException, clean_memory

logger = logging.getLogger(__name__)


def force_filesystem_sync():
    """
    Force filesystem sync and garbage collection to ensure disk usage stats are updated.
    This helps the Kaggle monitor reflect freed space more quickly.
    """
    # Flush filesystem buffers (Linux)
    os.system("sync")
    # Force Python garbage collection
    gc.collect()
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Small delay to allow system to update
    time.sleep(0.5)


def remove_safe(path: Path) -> None:
    """
    Remove a file, and if it's a hard link to a blob in the HF cache,
    also remove the blob when it becomes unreferenced.
    Logs the action.
    """
    if not path.exists():
        return

    # Locate the blob directory (HF cache structure)
    # path is like .../hub/models--repo--id/snapshots/hash/filename
    # blob directory is at .../hub/blobs
    hub_root = path.parent.parent.parent
    blob_dir = hub_root / "blobs"
    blob_file = None

    # Get inode of the snapshot file
    try:
        stat_before = path.stat()
        inode = stat_before.st_ino
    except OSError:
        # If we can't stat, just delete the file
        path.unlink()
        tqdm.write(f"  Removed {path.name} (could not stat, blob not tracked)")
        return

    # Find the corresponding blob file (if any) by inode
    if blob_dir.exists():
        for f in blob_dir.iterdir():
            if f.is_file():
                try:
                    if f.stat().st_ino == inode:
                        blob_file = f
                        break
                except OSError:
                    continue

    # Now delete the snapshot file
    path.unlink()
    tqdm.write(f"  Deleted snapshot file: {path.name}")

    # If we found a blob file, check its link count after deletion
    if blob_file is not None and blob_file.exists():
        try:
            # After deleting the snapshot, the blob's link count should be 1 if no other snapshots reference it
            if blob_file.stat().st_nlink == 1:
                blob_file.unlink()
                tqdm.write(f"  Deleted orphaned blob: {blob_file.name}")
        except OSError:
            pass

    # Force filesystem sync after deletion
    force_filesystem_sync()


def cleanup_orphaned_blobs(cache_root: Path) -> None:
    """
    Scan the blobs directory and delete any file with link count 1 (orphaned).
    """
    blob_dir = cache_root / "blobs"
    if not blob_dir.exists():
        return

    removed = 0
    freed = 0
    for f in blob_dir.iterdir():
        if f.is_file():
            try:
                if f.stat().st_nlink == 1:
                    size = f.stat().st_size
                    f.unlink()
                    removed += 1
                    freed += size
            except OSError:
                continue

    if removed > 0:
        tqdm.write(f"Cleaned up {removed} orphaned blobs, freed {freed / 1024**3:.2f} GB")
        force_filesystem_sync()


def log_disk_usage(cache_root: Path, checkpoint_path: Path, saving_path: Path) -> None:
    """
    Log current disk usage: blob directory size and total HF cache size.
    Called after each shard deletion.
    """
    # Ensure filesystem is synced before reading stats
    force_filesystem_sync()

    blob_dir = cache_root / "blobs"
    blob_size = 0
    if blob_dir.exists():
        blob_size = sum(f.stat().st_size for f in blob_dir.glob("*") if f.is_file()) / 1024**3

    # Also compute total cache size (hub directory)
    cache_size = 0
    if cache_root.exists():
        cache_size = sum(f.stat().st_size for f in cache_root.rglob("*") if f.is_file()) / 1024**3

    # Remaining shards in snapshot
    remaining_shards = list(Path(checkpoint_path).glob("*.safetensors")) + list(
        Path(checkpoint_path).glob("*.bin")
    )
    shard_count = len(remaining_shards)
    shard_size = sum(f.stat().st_size for f in remaining_shards) / 1024**3

    # Split layers size
    split_files = list(saving_path.glob("*.safetensors"))
    split_size = sum(f.stat().st_size for f in split_files) / 1024**3

    tqdm.write(
        f"--- Disk Usage After Deletion ---\n"
        f"  Remaining shards: {shard_count} files, {shard_size:.2f} GB\n"
        f"  Split layers: {len(split_files)} files, {split_size:.2f} GB\n"
        f"  Blobs directory: {blob_size:.2f} GB\n"
        f"  Total HF cache: {cache_size:.2f} GB"
    )


def check_space(
    checkpoint_path: Union[Path, str],
    layer_shards_saving_path: Optional[Union[Path, str]] = None,
    compression: Optional[str] = None,
    splitted_model_dir_name: str = "splitted_model",
) -> None:
    checkpoint_path = Path(checkpoint_path)
    total_shard_files_size_bytes = 0
    for model_shard_file in glob(str(checkpoint_path / "*")):
        total_shard_files_size_bytes += os.path.getsize(model_shard_file)

    total_saved_split_files_size_bytes = 0
    if layer_shards_saving_path is not None:
        for saved_split_file in glob(
            str(Path(layer_shards_saving_path) / splitted_model_dir_name / "*")
        ):
            total_saved_split_files_size_bytes += os.path.getsize(saved_split_file)

    if compression == "4bit":
        total_shard_files_size_bytes = int(total_shard_files_size_bytes * 0.2813)
    elif compression == "8bit":
        total_shard_files_size_bytes = total_shard_files_size_bytes // 2

    import shutil

    total, used, free = shutil.disk_usage(
        checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path
    )

    if free + total_saved_split_files_size_bytes < total_shard_files_size_bytes:
        save_path = (
            checkpoint_path if layer_shards_saving_path is None else layer_shards_saving_path
        )
        raise NotEnoughSpaceException(
            f"Not enough space. Free space under {save_path}:"
            f" {free / 1024 / 1024 / 1024:.02f}GB."
            f" Model total size: {total_shard_files_size_bytes / 1024 / 1024 / 1024:.02f}GB."
            f" Existing space under {save_path} assuming can reuse:"
            f" {total_saved_split_files_size_bytes / 1024 / 1024 / 1024:.02f}GB."
        )


def load_layer(
    local_path: Union[Path, str],
    layer_name: str,
    profiling: bool = False,
    persister: Optional[Any] = None,
    decompress: bool = True,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], float]]:
    p = persister if persister is not None else ModelPersister.get_model_persister()
    layer_state_dict = p.load_model(layer_name, local_path)

    if profiling:
        t = time.process_time()

    to_return = uncompress_layer_state_dict(layer_state_dict) if decompress else layer_state_dict

    if profiling:
        elapsed_time = time.process_time() - t
        return to_return, elapsed_time
    else:
        return to_return


def split_and_save_layers(
    checkpoint_path: Union[Path, str],
    layer_shards_saving_path: Optional[Union[Path, str]] = None,
    splitted_model_dir_name: str = "splitted_model",
    compression: Optional[str] = None,
    layer_names: Optional[Dict[str, str]] = None,
    delete_original: bool = False,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    hf_token: Optional[str] = None,
    sequential_shard_processing: bool = False,
    extreme_disk_cleanup: bool = False,
) -> str:
    logger.setLevel(logging.DEBUG)

    _token = token if token is not None else hf_token

    if compression is not None:
        assert bitsandbytes_installed, "when using compression bitsandbytes has to be installed."
        splitted_model_dir_name = splitted_model_dir_name + "." + compression

    checkpoint_path = Path(checkpoint_path)
    saving_path = checkpoint_path / splitted_model_dir_name
    if layer_shards_saving_path is not None:
        saving_path = Path(layer_shards_saving_path) / splitted_model_dir_name

    safetensors_format = False
    single_file_model = False
    if os.path.exists(checkpoint_path / "pytorch_model.bin.index.json"):
        with open(checkpoint_path / "pytorch_model.bin.index.json", "rb") as f:
            index = json.load(f)["weight_map"]
    elif os.path.exists(checkpoint_path / "model.safetensors.index.json"):
        safetensors_format = True
        with open(checkpoint_path / "model.safetensors.index.json", "rb") as f:
            index = json.load(f)["weight_map"]
    elif os.path.exists(checkpoint_path / "model.safetensors"):
        from safetensors import safe_open

        safetensors_format = True
        single_file_model = True
        single_file_path = checkpoint_path / "model.safetensors"
        with safe_open(str(single_file_path), framework="pt") as f:
            all_keys = f.keys()
        index = {k: "model.safetensors" for k in all_keys}
        del all_keys
    else:
        raise FileNotFoundError(
            f"No model checkpoint found in {checkpoint_path}. Expected one of: "
            "pytorch_model.bin.index.json, model.safetensors.index.json, or model.safetensors"
        )

    if layer_names is None:
        n_layers = len(set([int(k.split(".")[2]) for k in index.keys() if "model.layers" in k]))
    else:
        n_layers = len(
            set(
                [
                    int(k[len(layer_names["layer_prefix"]) :].split(".")[1])
                    for k in index.keys()
                    if layer_names["layer_prefix"] in k
                ]
            )
        )

    if layer_names is None:
        layers = (
            ["model.embed_tokens."]
            + [f"model.layers.{i}." for i in range(n_layers)]
            + ["model.norm.", "lm_head."]
        )
    else:
        layers = (
            [layer_names["embed"]]
            + [f"{layer_names['layer_prefix']}.{i}" for i in range(n_layers)]
            + [layer_names["norm"], layer_names["lm_head"]]
        )

        if "rotary_pos_emb" in layer_names:
            layers = [layer_names["rotary_pos_emb"]] + layers
        layers = [name + "." for name in layers]

    if os.path.exists(saving_path):
        found_layers = {}
        for layer in layers:
            found_layers[layer] = ModelPersister.get_model_persister().model_persist_exist(
                layer, saving_path
            )

        logger.debug("found_layers: %s", found_layers)
        if all(found_layers.values()):
            logger.info("saved layers already found in %s", saving_path)
            return str(saving_path)
        else:
            logger.warning(
                "some layer splits found, some are not, re-save all layers"
                " in case there's some corruptions."
            )

    if not delete_original:
        check_space(
            checkpoint_path,
            layer_shards_saving_path,
            compression,
            splitted_model_dir_name=splitted_model_dir_name,
        )

    shard = 0
    n_shards = len(set(index.values()))
    state_dict = {}

    if not os.path.exists(saving_path):
        saving_path.mkdir(parents=True, exist_ok=True)

    single_modelfile = None

    if single_file_model:
        single_modelfile = "model.safetensors"
        logger.info("Loading single-file model: %s", single_modelfile)
        state_dict = load_file(single_file_path, device="cpu")

    if sequential_shard_processing and not single_file_model:
        tqdm.write("===== ENTERING SEQUENTIAL SHARD PROCESSING MODE =====")
        tqdm.write(f"Compression setting: {compression}")
        layer_to_shards = {}
        shard_to_layers = {}

        tqdm.write("Building layer-to-shards and shard-to-layers from index...")
        for k, v in index.items():
            matched_layer = None
            for layer in layers:
                if k.startswith(layer):
                    matched_layer = layer
                    break
            if matched_layer:
                if matched_layer not in layer_to_shards:
                    layer_to_shards[matched_layer] = set()
                layer_to_shards[matched_layer].add(v)

                if v not in shard_to_layers:
                    shard_to_layers[v] = set()
                shard_to_layers[v].add(matched_layer)
            else:
                tqdm.write(f"Key {k} did not match any known layer prefix. It will be ignored.")

        sorted_shards = sorted(list(set(index.values())))
        processed_shards = set()
        partial_layers = {}
        total_layers = len(layer_to_shards)
        layers_processed = 0

        with tqdm(total=total_layers, desc="Assembling layers sequentially") as pbar:
            for shard_idx, shard_file in enumerate(sorted_shards, 1):
                tqdm.write(f"Loading shard {shard_idx}/{len(sorted_shards)}: {shard_file}")
                to_load = checkpoint_path / shard_file
                if not os.path.exists(to_load):
                    assert repo_id is not None
                    tqdm.write(f"Shard {shard_file} not found locally, downloading...")
                    huggingface_hub.snapshot_download(
                        repo_id, allow_patterns=os.path.basename(to_load), token=_token
                    )

                if not safetensors_format:
                    shard_state_dict = torch.load(to_load, map_location="cpu")
                else:
                    shard_state_dict = load_file(to_load, device="cpu")

                for k, v in shard_state_dict.items():
                    matched_layer = None
                    for layer in layers:
                        if k.startswith(layer):
                            matched_layer = layer
                            break
                    if matched_layer:
                        if matched_layer not in partial_layers:
                            partial_layers[matched_layer] = {}
                        partial_layers[matched_layer][k] = v

                del shard_state_dict

                if delete_original and (shard_file not in shard_to_layers or len(shard_to_layers[shard_file]) == 0):
                    to_delete = checkpoint_path / shard_file
                    if os.path.exists(to_delete):
                        remove_safe(to_delete)
                        # After deletion, log disk usage
                        hub_root = checkpoint_path.parent.parent.parent
                        log_disk_usage(hub_root, checkpoint_path, saving_path)
                    if shard_file in shard_to_layers:
                        del shard_to_layers[shard_file]

                processed_shards.add(shard_file)

                completed_layers = []
                for layer, required_shards in layer_to_shards.items():
                    if required_shards.issubset(processed_shards):
                        completed_layers.append(layer)

                for layer in completed_layers:
                    shards_for_this_layer = layer_to_shards[layer]
                    del layer_to_shards[layer]

                    tqdm.write(f"Layer {layer} is now complete. Required shards: {shards_for_this_layer}")

                    if layer in partial_layers:
                        layer_state_dict = partial_layers.pop(layer)

                        # --- Calculate uncompressed size ---
                        uncompressed_bytes = sum(
                            v.numel() * v.element_size() for v in layer_state_dict.values()
                        )
                        uncompressed_gb = uncompressed_bytes / 1024**3

                        # --- Compress and save ---
                        layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)
                        marker_exists = ModelPersister.get_model_persister().model_persist_exist(
                            layer, saving_path
                        )
                        if not marker_exists:
                            ModelPersister.get_model_persister().persist_model(
                                layer_state_dict, layer, saving_path
                            )

                        # --- Get saved file size ---
                        layer_filename = layer.rstrip('.') + ".safetensors"
                        layer_file = saving_path / layer_filename
                        if layer_file.exists():
                            saved_bytes = layer_file.stat().st_size
                            saved_gb = saved_bytes / 1024**3
                            ratio = saved_bytes / uncompressed_bytes if uncompressed_bytes > 0 else 0
                            tqdm.write(
                                f"  Compressed {layer}: original {uncompressed_gb:.3f} GB, "
                                f"saved {saved_gb:.3f} GB (ratio {ratio:.2f})"
                            )
                        else:
                            tqdm.write(f"  WARNING: Saved layer file not found: {layer_file}")

                        del layer_state_dict
                        layers_processed += 1
                        pbar.update(1)
                    else:
                        tqdm.write(f"WARNING: Layer {layer} was completed but had no state dict keys!")
                        pbar.update(1)

                    if delete_original:
                        for s_file in shards_for_this_layer:
                            if s_file in shard_to_layers and layer in shard_to_layers[s_file]:
                                shard_to_layers[s_file].remove(layer)
                                remaining = shard_to_layers[s_file]
                                tqdm.write(f"Shard {s_file} now has remaining layers: {remaining}")
                                if len(remaining) == 0:
                                    to_delete = checkpoint_path / s_file
                                    if os.path.exists(to_delete):
                                        remove_safe(to_delete)
                                        # After deletion, log disk usage
                                        hub_root = checkpoint_path.parent.parent.parent
                                        log_disk_usage(hub_root, checkpoint_path, saving_path)
                                    del shard_to_layers[s_file]

                clean_memory()

        # At the end, process leftovers (if any)
        for layer, layer_state_dict in list(partial_layers.items()):
            tqdm.write(f"Processing leftover partial layer: {layer}")
            uncompressed_bytes = sum(v.numel() * v.element_size() for v in layer_state_dict.values())
            uncompressed_gb = uncompressed_bytes / 1024**3

            layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)
            marker_exists = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)
            if not marker_exists:
                ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)

            layer_filename = layer.rstrip('.') + ".safetensors"
            layer_file = saving_path / layer_filename
            if layer_file.exists():
                saved_bytes = layer_file.stat().st_size
                saved_gb = saved_bytes / 1024**3
                ratio = saved_bytes / uncompressed_bytes if uncompressed_bytes > 0 else 0
                tqdm.write(
                    f"  Compressed leftover {layer}: original {uncompressed_gb:.3f} GB, "
                    f"saved {saved_gb:.3f} GB (ratio {ratio:.2f})"
                )

            del layer_state_dict

            if delete_original:
                for s_file in list(shard_to_layers.keys()):
                    if layer in shard_to_layers[s_file]:
                        shard_to_layers[s_file].remove(layer)
                        if len(shard_to_layers[s_file]) == 0:
                            to_delete = checkpoint_path / s_file
                            if os.path.exists(to_delete):
                                remove_safe(to_delete)
                                hub_root = checkpoint_path.parent.parent.parent
                                log_disk_usage(hub_root, checkpoint_path, saving_path)
                            del shard_to_layers[s_file]

        partial_layers.clear()

        if delete_original:
            for s_file, remaining_layers in list(shard_to_layers.items()):
                if len(remaining_layers) > 0:
                    tqdm.write(
                        f"WARNING: Shard {s_file} still has {len(remaining_layers)} layers "
                        f"preventing its deletion! Sample: {list(remaining_layers)}"
                    )

            for s_file in list(shard_to_layers.keys()):
                to_delete = checkpoint_path / s_file
                if os.path.exists(to_delete):
                    remove_safe(to_delete)
                    hub_root = checkpoint_path.parent.parent.parent
                    log_disk_usage(hub_root, checkpoint_path, saving_path)
                del shard_to_layers[s_file]

            remaining_files = list(Path(checkpoint_path).glob("*.safetensors")) + list(
                Path(checkpoint_path).glob("*.bin")
            )
            if remaining_files:
                tqdm.write(
                    f"WARNING: After processing, some shard files still exist: "
                    f"{[f.name for f in remaining_files]}"
                )
            else:
                tqdm.write("All shard files have been deleted.")

        # --- Extreme disk cleanup ---
        if extreme_disk_cleanup:
            tqdm.write("===== EXTREME DISK CLEANUP ACTIVATED =====")
            # Delete the entire snapshot directory (original model files)
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                tqdm.write(f"Deleted entire snapshot directory: {checkpoint_path}")
                force_filesystem_sync()
            # Clean up orphaned blobs globally
            hub_root = checkpoint_path.parent.parent.parent
            cleanup_orphaned_blobs(hub_root)
            tqdm.write("Extreme cleanup completed.")

        # Final disk usage summary
        hub_root = checkpoint_path.parent.parent.parent
        force_filesystem_sync()  # Ensure latest stats
        blob_dir = hub_root / "blobs"
        blob_size = 0
        if blob_dir.exists():
            blob_size = sum(f.stat().st_size for f in blob_dir.glob("*") if f.is_file()) / 1024**3

        final_shards = list(Path(checkpoint_path).glob("*.safetensors")) + list(
            Path(checkpoint_path).glob("*.bin")
        )
        final_split = list(saving_path.glob("*.safetensors"))
        tqdm.write(
            "===== FINAL DISK USAGE =====\n"
            f"Shards remaining: {len(final_shards)} files, "
            f"{sum(f.stat().st_size for f in final_shards) / 1024**3:.2f} GB\n"
            f"Split layers: {len(final_split)} files, "
            f"{sum(f.stat().st_size for f in final_split) / 1024**3:.2f} GB\n"
            f"Blobs directory: {blob_size:.2f} GB\n"
            f"Total: {(sum(f.stat().st_size for f in final_shards) + sum(f.stat().st_size for f in final_split) + (sum(f.stat().st_size for f in blob_dir.glob('*') if f.is_file()) if blob_dir.exists() else 0)) / 1024**3:.2f} GB"
        )

        clean_memory()

    else:
        # Original non‑sequential branch (unchanged)
        for layer in tqdm(layers):
            if not single_file_model:
                shards = [
                    int(v.split("-")[1])
                    for k, v in index.items()
                    if k.startswith(layer) and "-" in v and len(v.split("-")) > 1
                ]
                if len(shards) > 0:
                    if max(shards) > shard:
                        if delete_original and shard != 0:
                            if not safetensors_format:
                                to_delete = (
                                    checkpoint_path
                                    / f"pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin"
                                )
                            else:
                                to_delete = (
                                    checkpoint_path
                                    / f"model-000{shard:02d}-of-000{n_shards:02d}.safetensors"
                                )

                            logger.info("Deleted shard: %s", to_delete)
                            remove_safe(to_delete)
                            hub_root = checkpoint_path.parent.parent.parent
                            log_disk_usage(hub_root, checkpoint_path, saving_path)
                        shard += 1
                        logger.info("Loading shard %s/%s", shard, n_shards)

                        if not safetensors_format:
                            to_load = (
                                checkpoint_path
                                / f"pytorch_model-000{shard:02d}-of-000{n_shards:02d}.bin"
                            )
                        else:
                            to_load = (
                                checkpoint_path
                                / f"model-000{shard:02d}-of-000{n_shards:02d}.safetensors"
                            )

                        if not os.path.exists(to_load):
                            assert repo_id is not None
                            huggingface_hub.snapshot_download(
                                repo_id, allow_patterns=os.path.basename(to_load), token=_token
                            )

                        if not safetensors_format:
                            state_dict.update(torch.load(to_load, map_location="cpu"))
                        else:
                            state_dict.update(load_file(to_load, device="cpu"))

                else:
                    shards = [v for k, v in index.items() if k.startswith(layer)]
                    single_modelfile = shards[0]
                    to_load = checkpoint_path / single_modelfile
                    if not os.path.exists(to_load):
                        assert repo_id is not None
                        huggingface_hub.snapshot_download(
                            repo_id, allow_patterns=os.path.basename(to_load), token=_token
                        )
                    if not safetensors_format:
                        state_dict.update(torch.load(to_load, map_location="cpu"))
                    else:
                        state_dict.update(load_file(to_load, device="cpu"))

            layer_state_dict = dict([(k, v) for k, v in state_dict.items() if k.startswith(layer)])

            layer_state_dict = compress_layer_state_dict(layer_state_dict, compression)

            marker_exists = ModelPersister.get_model_persister().model_persist_exist(layer, saving_path)
            if not marker_exists:
                ModelPersister.get_model_persister().persist_model(layer_state_dict, layer, saving_path)

            for k in layer_state_dict.keys():
                if k in state_dict:
                    del state_dict[k]
            del layer_state_dict
            clean_memory()

    if delete_original and single_modelfile is not None:
        to_delete = checkpoint_path / single_modelfile
        logger.info("Deleted original file: %s", to_delete)
        remove_safe(to_delete)
        hub_root = checkpoint_path.parent.parent.parent
        log_disk_usage(hub_root, checkpoint_path, saving_path)

    return str(saving_path)


def find_or_create_local_splitted_path(
    model_local_path_or_repo_id: str,
    layer_shards_saving_path: Optional[Union[Path, str]] = None,
    compression: Optional[str] = None,
    layer_names: Optional[Dict[str, str]] = None,
    token: Optional[str] = None,
    hf_token: Optional[str] = None,
    delete_original: bool = False,
    sequential_shard_processing: bool = False,
    extreme_disk_cleanup: bool = False,
) -> Tuple[Path, str]:
    _token = token if token is not None else hf_token

    if os.path.exists(model_local_path_or_repo_id):
        has_index = os.path.exists(
            Path(model_local_path_or_repo_id) / "pytorch_model.bin.index.json"
        ) or os.path.exists(Path(model_local_path_or_repo_id) / "model.safetensors.index.json")
        has_single_file = os.path.exists(Path(model_local_path_or_repo_id) / "model.safetensors")
        if has_index or has_single_file:
            logger.info("found model checkpoint...")
            return Path(model_local_path_or_repo_id), split_and_save_layers(
                model_local_path_or_repo_id,
                layer_shards_saving_path,
                compression=compression,
                layer_names=layer_names,
                delete_original=delete_original,
                sequential_shard_processing=sequential_shard_processing,
                extreme_disk_cleanup=extreme_disk_cleanup,
            )
        else:
            logger.warning(
                "Found local directory in %s, but didn't find downloaded model."
                " Try using it as a HF repo...",
                model_local_path_or_repo_id,
            )

    hf_cache_path = huggingface_hub.snapshot_download(
        model_local_path_or_repo_id, token=_token, ignore_patterns=["*.safetensors", "*.bin"]
    )

    has_index = os.path.exists(
        Path(hf_cache_path) / "pytorch_model.bin.index.json"
    ) or os.path.exists(Path(hf_cache_path) / "model.safetensors.index.json")
    if not has_index:
        hf_cache_path = huggingface_hub.snapshot_download(
            model_local_path_or_repo_id, token=_token, allow_patterns=["model.safetensors"]
        )

    return Path(hf_cache_path), split_and_save_layers(
        hf_cache_path,
        layer_shards_saving_path,
        compression=compression,
        layer_names=layer_names,
        delete_original=delete_original,
        repo_id=model_local_path_or_repo_id,
        token=_token,
        sequential_shard_processing=sequential_shard_processing,
        extreme_disk_cleanup=extreme_disk_cleanup,
    )