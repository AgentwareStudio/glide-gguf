# Native runtime

Use a POSIX/Linux environment with Python 3.12+, CMake >=3.20, a C++17 compiler, and the appropriate CUDA toolkit for a CUDA build. This package does not contain prebuilt executables. 

From the cloned GitHub repository, download the model and tokenizer assets from Hugging Face into this directory:

```sh
python -m pip install huggingface_hub
hf download Agentware/GLiDE-GGUF --include "glide-Q4_K_S-q8emb-bf16restore.gguf" "upstream/tokenizer.json" "upstream/readout.safetensors" --local-dir .
```

The GitHub repository contains the runtime and configuration; large assets remain on Hugging Face. `MODEL_SHA256SUMS` records the original model release hashes, while `SHA256SUMS` covers this code repository.

From this directory, create an isolated Python environment and install `requirements-tokenizer.txt`. These versions were checked for tokenizer and prompt construction. PyTorch is not needed for this native-wrapper tokenizer-only path.

Fetch the exact official llama.cpp source and verify all files (no model inference):
```sh
python native/runtime/fetch_llama.py /absolute/path/to/llama.cpp
python native/runtime/verify_llama_source.py /absolute/path/to/llama.cpp
```
The fetcher refuses an existing destination and validates the archive SHA256. CMake also validates the 3,901-file source manifest.

For an NVIDIA L4-targeted build, the architecture is 89. Build commands:
```sh
cmake -S native/runtime -B build-cuda \
  -DLLAMA_CPP_DIR=/absolute/path/to/llama.cpp \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda --target glide-logits-fa -j2
```

Use only the dedicated wrapper and original input schema. From the release root:
```python
import os, sys
os.environ['GGML_CUDA_NO_PINNED'] = '1'
sys.path.insert(0, 'native/python')
from glide_gguf import GlideGGUF
with GlideGGUF.from_pretrained(
    'upstream', gguf_path='glide-Q4_K_S-q8emb-bf16restore.gguf',
    runner_path='build-cuda/glide-logits-fa', context_size=8192,
    threads=4, gpu_layers=-1, timeout_seconds=600,
) as model:
    result = model.decide(
        'The user asks to reset a password.',
        {'category': {'type': 'choice', 'instructions': 'Choose a category.',
                      'criteria': {'account': 'Account assistance', 'other': 'Other'}}},
    )
    print(result)
```
Each question is processed serially with full noncausal attention, a state reset, raw F32 readout and original softmax. Images, partial GPU layer offload and oversized prompts are rejected; no silent truncation.

See the model card for the measured memory use and the remaining cross-backend probability differences.
