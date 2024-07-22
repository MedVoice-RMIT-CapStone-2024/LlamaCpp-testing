# Quick Instruction for LlamaCpp setup (Linux)

After creating virtual environment and install `langchain`:

1. Run this command to use GPU version of LlamaCpp (require cmake-3.29.6 refer to this [link](https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line)):
```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```
2. Access the llama guard 2 gguf modelfile and download it via this [link](https://rmiteduau-my.sharepoint.com/:f:/g/personal/s3924826_rmit_edu_vn/EpoVJIgorNNDiB7FTABtR-QBuO3zk5GPbD-DsKUgR8Ggeg?e=5uXcwL). This model has been quantized to `Q4_K_M` for the ease of use. Or you can go to hugging face and look up for gguf-file models then copy the link ``` wget "the copied link"```

    Now, you are ready to run the test.py for demonstration!

**Notice that the LlamaCpp implementation for llama guard 2 modelfile is from line 20 to line 73!**
