set source_file=%1
set target_file=%2

ECHO "You need to set up the paths for your system! Comment this line when done."

:: Example command
::call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64


::"cl" "/LD" %source_file%
