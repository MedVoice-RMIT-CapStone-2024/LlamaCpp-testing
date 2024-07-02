# Quick Instruction for LlamaCpp setup (Linux)

After creating virtual environment and install `langchain`:

1. Run this command to use GPU version of LlamaCpp:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```
2. Access the llama guard 2 gguf modelfile and download it via this [link](https://rmiteduau-my.sharepoint.com/:f:/g/personal/s3924826_rmit_edu_vn/EpoVJIgorNNDiB7FTABtR-QBuO3zk5GPbD-DsKUgR8Ggeg?e=5uXcwL). This model has been quantized to `Q4_K_M` for the ease of use.

    Now, you are ready to run the test.py for demonstration!

**Notice that the LlamaCpp implementation for llama guard 2 modelfile is from line 20 to line 73!**
