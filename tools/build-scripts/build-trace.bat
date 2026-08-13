@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\CMake\bin\cmake.exe" --build "C:\Users\yassi\Documents\code\BeyondVram\tools\llama.cpp-source\build" --config Release --target llama-moe-trace
