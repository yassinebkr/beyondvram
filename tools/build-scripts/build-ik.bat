@echo off
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=llama-bench"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\CMake\bin\cmake.exe" --build "C:\Users\yassi\Documents\code\BeyondVram\tools\ik_llama.cpp\build" --config Release --target %TARGET%
