# Product Requirements Document: Sequential Shard Processing for RabbitLLM

## 1. Overview
**Feature Name:** Sequential Shard Processing  
**Objective:** Enable RabbitLLM to load, split, compress, and save model layers **one shard at a time**, deleting each original shard immediately after its tensors have been processed. This reduces peak disk space usage from `original_model_size + compressed_size` to approximately `max_shard_size + compressed_size`, allowing very large models to be processed on disk-constrained environments like Kaggle notebooks.

**Problem:** Currently, RabbitLLM downloads all shards of a model, then splits them into per‑layer files. During this process, both the original shards and the newly created layer files coexist, requiring disk space equal to the sum of both. For a 150 GB model, this peak can exceed 190 GB, which is far beyond Kaggle’s ~60 GB temporary storage. Users who need to run large models (70B+) on limited hardware are unable to do so.

**Solution:** Process shards sequentially: for each shard, load its tensors, assign them to the appropriate layer buffers, and once a layer is complete (all its tensors collected), write the layer file (with optional compression) and free the layer buffer. Immediately after processing a shard, delete that shard file. This way, at any moment, the disk holds at most one original shard plus the final compressed layer files (which accumulate). The peak disk usage becomes the size of the largest shard plus the total compressed layer size.

**Benefits:**
- Drastic reduction in peak disk footprint.
- Enables running models like Qwen3-Coder-Next-72B (original 150 GB, shards ~5 GB each) with compression on Kaggle.
- Maintains all existing RabbitLLM features (layer streaming, compression, profiling).

## 2. API Change
Add a new boolean parameter to `AutoModel.from_pretrained`:

- **Parameter name:** `sequential_shard_processing`
- **Type:** `bool`
- **Default:** `False` (preserves current behavior: all shards downloaded first, then split)
- **When `True`:** Activates the sequential shard processing logic described in this document.

**Location:** The parameter should be added to the `from_pretrained` method signature and passed down to the internal splitting routine. No other public API changes are required.

## 3. Detailed Functional Requirements

### 3.1 Pre‑processing: Parse Index File
Before starting the shard loop, the system must analyze the model’s index file (`model.safetensors.index.json`) to understand the relationship between tensors, layers, and shards.

- **Input:** Path to the Hugging Face model directory containing the index file.
- **Output:** Two data structures:
  - **Layer-to-shards mapping:** A dictionary that maps each layer identifier (e.g., `"layers.0"`, `"embed_tokens"`) to a set of shard filenames that contain tensors belonging to that layer.
  - **Layer-to-tensors mapping:** A dictionary that maps each layer identifier to a list of tensor names that make up that layer.
- **Implementation logic:**
  - Load the JSON index.
  - Iterate over each entry in the `weight_map`.
  - For each tensor name, extract a layer identifier using a configurable pattern (must support different model architectures). For Qwen2/3 and Llama families, typical patterns are:
    - `"model.layers.0.self_attn.q_proj.weight"` → `"layers.0"`
    - `"model.embed_tokens.weight"` → `"embed_tokens"`
    - `"model.norm.weight"` → `"norm"`
    - `"lm_head.weight"` → `"lm_head"`
  - Add the tensor name to the layer’s tensor list, and add the shard filename to the layer’s shard set.
  - Handle special cases like tied weights (where a tensor appears under multiple names) to avoid double-counting; typically the index lists each weight once, so no extra action is needed.

### 3.2 Sequential Shard Processing Loop
The core loop processes shards one by one in numerical order (e.g., `model-00001-of-00010.safetensors`, `model-00002-of-00010.safetensors`, …).

#### 3.2.1 Initialization
- Determine the list of shard files present in the model directory.
- Initialize a set `processed_shards` to keep track of which shards have been handled.
- Initialize a dictionary `partial_layers` to hold tensors for layers that are not yet complete (i.e., layers whose tensors span multiple shards). The structure is `{layer_id: {tensor_name: tensor}}`.

#### 3.2.2 For Each Shard
1. **Load the shard** using the appropriate safetensors loader. Load the entire shard into memory (CPU RAM).
2. **Process each tensor** in the shard:
   - Determine the tensor’s layer ID using the same extraction logic from pre‑processing.
   - Store the tensor in `partial_layers[layer_id][tensor_name]`.
3. **Immediately after processing all tensors in the shard, delete the shard file** from disk. This is the critical step for reducing peak disk usage.
4. **Update `processed_shards`** by adding the current shard filename.
5. **Check for layer completion:** For every layer ID currently in `partial_layers`, compare the set of shards it requires (from the pre‑built layer-to-shards mapping) with `processed_shards`. If `processed_shards` contains all required shards for that layer, then the layer is complete.
   - For each completed layer:
     - Retrieve the layer’s tensors from `partial_layers`.
     - If compression is enabled (`compression` parameter set to `"4bit"` or `"8bit"`), apply block‑wise quantization to the layer’s weight tensors. Use the existing RabbitLLM compression routines (which rely on `bitsandbytes`).
     - Write the layer’s tensors (compressed or original) to a safetensors file in the directory specified by `layer_shards_saving_path`. The filename should follow the existing naming convention (e.g., `layer_0000.safetensors`).
     - Remove the layer from `partial_layers` to free memory.
6. **Optional memory cleanup:** After each shard, explicitly delete the loaded shard data and call Python’s garbage collector if memory pressure is high.

#### 3.2.3 Post‑Processing
After all shards have been processed, the `partial_layers` dictionary should be empty because all layers should have been completed. However, as a safeguard, iterate over any remaining entries and process them as completed layers (writing them to disk). This handles any layers that may have been missed due to logic errors.

### 3.3 Handling Multi‑Shard Layers
- Some layers (e.g., large embedding matrices) may have their tensors split across multiple shards. The algorithm naturally handles this because `partial_layers` accumulates tensors from successive shards until the layer’s required shard set is complete.
- There is no need for special pre‑allocation; the buffer grows as tensors arrive. However, ensure that memory usage is monitored: a single layer could be very large (e.g., the embedding layer of a 70B model), but that layer’s tensors will only occupy memory once all its shards are loaded. This is acceptable as long as the total RAM (typically ~30 GB on Kaggle) is not exceeded.

### 3.4 Interaction with Existing Features
- **Compression:** If compression is requested, it is applied **after** all tensors for a layer have been assembled, just before writing the layer file. The existing quantization functions should be reused without modification.
- **`delete_original` flag:** This flag, which currently deletes the entire original Hugging Face cache after splitting, becomes partially redundant when sequential processing is enabled because shards are deleted during the loop. However, for consistency and backward compatibility, the flag should still be respected:
  - If `sequential_shard_processing=True` and `delete_original=True`, the shard‑by‑shard deletion still occurs; after the loop, the original model directory may be empty, and any remaining cache files (like the index) can be removed if the flag is `True`.
  - If `sequential_shard_processing=True` and `delete_original=False`, only the shard files are deleted (as part of the loop), but the rest of the cache (index, config) may be kept.
  - The logic should be implemented so that the two flags work independently and without conflict.
- **`layer_shards_saving_path`:** This parameter continues to specify where the final per‑layer files are written.
- **Profiling mode:** If `profiling_mode=True`, the system should still record and print per‑layer timing information, now including the time spent waiting for layers to complete across multiple shards.
- **Prefetching:** The existing prefetching mechanism (which loads the next layer while computing the current one) operates at the layer level and is independent of shard loading. Sequential shard processing only affects how layers are initially split and saved; during inference, the same layer streaming logic applies. Therefore, prefetching remains unchanged.

### 3.5 Error Handling and Resilience
- **Shard load failure:** If a shard fails to load (corrupted file, I/O error), the process should abort with a clear error message. The temporary layer files already written may be incomplete; the user should be instructed to clear the cache and retry.
- **Interruption during processing:** If the kernel or process dies in the middle of sequential shard processing, partially written layer files may exist. On the next run, the system should detect that some layers are already present and either skip them (if they are complete) or overwrite them. This requires idempotency: before writing a layer, check if the destination file already exists and if it is valid (e.g., by comparing sizes or using a checksum). If the user wants to restart fresh, they can manually delete the cache directory.
- **Disk space monitoring:** While not strictly required, it is advisable to add a check before processing each shard to ensure there is enough free space for the remaining compressed layers. If free space drops below a threshold (e.g., 1 GB), warn the user or abort to prevent kernel death.

### 3.6 Backward Compatibility
- When `sequential_shard_processing=False`, the existing code path must remain unchanged and fully functional.
- The new parameter should be optional and ignored by older versions of the library (if this is a new release, versioning should handle it).
- All existing unit tests should pass with the default setting.

## 4. Performance Considerations
- **Disk I/O:** Sequential processing reads each shard once and writes each layer once, exactly the same total I/O as the current method. No additional overhead.
- **Memory usage:** Peak RAM usage is the size of the largest shard (typically 5‑10 GB) plus any incomplete layer buffers. For a 70B model with large embedding layers that span multiple shards, the embedding layer buffer could be large (e.g., several GB). This is still within typical Kaggle RAM limits (30 GB). If necessary, a memory usage warning can be added when a layer’s total size exceeds a threshold.
- **CPU usage:** Slightly more CPU time may be spent on repeated layer‑completion checks, but this is negligible compared to the cost of quantization (if enabled) and I/O.
- **Time to first layer:** The first layer will be written only after all its shards are processed. For layers that span many shards, this could delay the availability of that layer. However, during inference, RabbitLLM streams layers in order, so the first layer (e.g., embedding) may be needed early. The existing logic should still work because the layer file will be available by the time inference needs it. The splitting process must complete fully before inference can begin; this is unchanged.

## 5. Testing Guidelines
To ensure the feature works correctly, the following test scenarios should be covered:

- **Unit tests (mocked):**
  - Test layer ID extraction for different architectures (Qwen2, Qwen3, Llama).
  - Test building layer‑to‑shards mapping from a mock index.
  - Test completion detection logic with various shard sets.
  - Verify that shard files are deleted after processing.
  - Verify that compressed layers are written correctly.
- **Integration tests with real small models:**
  - Run on a model like `Qwen/Qwen2.5-0.5B` with `sequential_shard_processing=True` and `compression="4bit"`. Compare the output files and inference results with the standard method.
  - Measure peak disk usage using a filesystem monitor to confirm reduction.
- **Edge cases:**
  - Model with a layer that spans multiple shards (e.g., a large embedding).
  - Model with tied weights (e.g., `lm_head` tied to `embed_tokens`). Ensure no duplication or missing tensors.
  - Interruption recovery: simulate a crash after writing some layers, then rerun and verify that remaining layers are correctly added without duplication.
  - Run with `sequential_shard_processing=True` and `delete_original=False`; verify that only shards are deleted, not the entire cache.
- **Performance tests:** On a reasonably sized model, measure runtime and memory usage compared to the standard method; ensure no significant regression.

## 6. Documentation Updates
- **README.md:** Add a section describing the new parameter, its purpose, and when to use it (especially for disk‑constrained environments). Provide a brief example.
- **docs/ARCHITECTURE.md:** Expand the “Model Splitting” section to include the sequential shard processing algorithm, explaining the trade‑offs and benefits.
- **docs/TROUBLESHOOTING.md:** Add notes about disk space issues and how the new feature can help. Also mention that if the process is interrupted, partial layers may remain and how to clean up.
- **CONTRIBUTING.md:** If the change requires modifications to the core splitting logic, update the developer guidelines.

## 7. Risks and Mitigations
| Risk | Mitigation |
|------|------------|
| **Layers that span many shards cause high memory usage while buffering.** | Monitor memory; if a layer’s total size exceeds, say, 10 GB, issue a warning but continue. This is still within Kaggle’s 30 GB limit. |
| **Layer ID extraction may fail for new or custom architectures.** | Use a fallback mechanism: if the architecture is not recognized, assume that tensor names follow a pattern like `"model.layers.<digit>.*"`. If that fails, log a warning and treat each tensor as its own layer (which is inefficient but safe). |
| **Disk space still insufficient if a single shard is larger than available temporary space.** | This is unlikely because shard sizes are typically 5‑10 GB. However, document that the user must have at least the size of the largest shard plus a buffer (e.g., 10 GB) free. |
| **Interruption leaves partial layers that are invalid or incomplete.** | Implement idempotent writes: before writing a layer, check if a file with the same name exists and if it contains the expected number of tensors. If the file exists and appears complete, skip writing. Provide a manual cleanup instruction. |
| **Performance regression due to repeated completion checks.** | Optimize by only checking layers that received new tensors in the current shard, not all layers every time. The completion check should be O(1) per layer using set operations. |

## 8. Appendix: High‑Level Algorithm Description (Pseudocode)
For implementers, a step‑by‑step outline of the algorithm (without actual code):

1. **Parse index:** Load `model.safetensors.index.json`. Create a dictionary mapping each layer identifier to the set of shard files it depends on, and a list of tensor names per layer.
2. **List shards:** Get sorted list of all `.safetensors` shard files in the model directory.
3. **Initialize:** `processed_shards = set()`, `partial_layers = {}` (dictionary of layer → dict of tensors).
4. **For each shard in sorted list:**
   a. Load shard into memory.
   b. For each tensor in shard:
      - Determine layer ID (using same logic as step 1).
      - If layer not in `partial_layers`, initialize an empty dict.
      - Add tensor to `partial_layers[layer]`.
   c. Delete shard file from disk.
   d. Add shard filename to `processed_shards`.
   e. Identify layers that are now complete:
      - For each layer ID in `partial_layers`:
        - If `layer_to_shards[layer]` is a subset of `processed_shards`:
          - If compression is enabled, quantize all tensors in `partial_layers[layer]`.
          - Write the layer file to `layer_shards_saving_path`.
          - Remove layer ID from `partial_layers`.
   f. Optionally free memory: `del shard_data`, `gc.collect()`.
5. **After loop:** Any remaining layers in `partial_layers` should be complete (by invariant). Process them as above.
6. **Clean up:** If `delete_original=True`, remove any remaining files in the original cache directory (e.g., the index file, config files). Otherwise, leave them.

This algorithm ensures that at most one shard is on disk at any time, and layer files are written incrementally, achieving the desired disk space reduction.