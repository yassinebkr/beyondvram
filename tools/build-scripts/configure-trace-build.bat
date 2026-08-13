@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin;%PATH%"
"C:\Program Files\CMake\bin\cmake.exe" -G Ninja -DCMAKE_MAKE_PROGRAM="C:\Users\yassi\Documents\code\BeyondVram/.venv/Scripts/ninja.exe" -S "C:\Users\yassi\Documents\code\BeyondVram/tools/llama.cpp-source" -B "C:\Users\yassi\Documents\code\BeyondVram/tools/llama.cpp-source/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
