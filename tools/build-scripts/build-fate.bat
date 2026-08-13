@echo off
rem Configure + build the llama-moe-cache (FATE) fork. AGPL-3.0 third-party
rem code; built for the reproduce-or-falsify experiment only (results go to
rem results/moe-locality/fate-repro/). Same toolchain path as build-trace.bat.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "PATH=%CUDA_PATH%\bin;%PATH%"
"C:\Program Files\CMake\bin\cmake.exe" -G Ninja -DCMAKE_MAKE_PROGRAM="C:\Users\yassi\Documents\code\BeyondVram/.venv/Scripts/ninja.exe" -S "C:\Users\yassi\Documents\code\BeyondVram/tools/llama-moe-cache" -B "C:\Users\yassi\Documents\code\BeyondVram/tools/llama-moe-cache/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF
if errorlevel 1 exit /b 1
"C:\Program Files\CMake\bin\cmake.exe" --build "C:\Users\yassi\Documents\code\BeyondVram\tools\llama-moe-cache\build" --config Release --target llama-completion llama-bench llama-cli
