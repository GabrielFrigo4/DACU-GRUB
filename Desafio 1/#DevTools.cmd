@echo off
set msvc_arch=x64
if [%1]==[x86] set msvc_arch=x86
if [%1]==[x64] set msvc_arch=x64

set VSCMD_SKIP_SENDTELEMETRY=1
cmd /k "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=%msvc_arch% -host_arch=%msvc_arch%
