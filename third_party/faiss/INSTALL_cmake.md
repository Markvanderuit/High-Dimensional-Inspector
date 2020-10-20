# Motivation
I am not satisfied that faiss developers do not provide window support, so I came up with this faiss branch myself.
I do not intend to add any new features or algorithms in this branch, the pure purpose of this branch is to make faiss more cross-platform.
Therefore I adopt cmake to build faiss. Actually cmake files were in faiss repository 2 years ago in 2018, i do not know why they abandoned them later.
Note that this is an on-going effort, bug reports are welcome.

# current status and known issues
* Only 64bit build is supported
* OnDiskInvertedList is not supported on Windows yet

# compilation and build guide
## requirements
cmake 3.15 and above is required.
Tested on
* Windows 10, Visual Studio Community Edition 2017/2019, CUDA 10.2
* Ubuntu 19.10, GCC 8/9 , CUDA 10.2
MacOS not tested
## build faiss
faiss requires a BLAS library, Intel MKL is free and to my best knowledge, one of the best implementations in terms of performance.
So in this guide we assume that Intel MKL SDK is pre-installed.
1. open cmake-gui, press configure, it will report errors, because it can not find any BLAS library by default.
1. check 'WITH-MKL', and in the field `MKL_INCLUDE_DIRS` input the intel MKL include file directory, then configure again 
    1. you are free to check `BUILD_TEST`,`BUILD_TUTORIAL`. If you want to build all the test cases, make sure you git clone the code with recursive flag so that external repository `gtest` is also pulled.
	1. do not check `BUILD_WITH_GPU` if cuda is not available.
1. generate 
    1. on windows you can build all the pojects with visual studio 
	1. on Linux, go to the folder where you generate make file,and do make
## build gpu faiss
building gpu faiss is pretty much the same, just ensure that CUDA is installed, and check `BUILD_WITH_GPU`, the rest is taken care of by cmake
## build python wrapper

### Windows
make  sure swig, python and numpy are already installed
1. open visual stdudio 2017/2019 development console
1. go to  $(faiss_root_dir)/python
1. generate swig wrapper cxx file
    1. for cpu faiss, run `swig -python -c++ -I../../ swigfaiss.swig`
    1. gor gpu faiss, run `swig -python -c++ -DGPU_WRAPPER -I../../ swigfaiss.swig`
1. setting environment variables by running `set PythonHome = E:\Anaconda3` (change it accordingly)
   `set PythonLibName = python36.lib` (change it accordingly if you have a different version of python)
   `set MKL_INCLUDE_DIRS = E:/sdk/Intel/MKL/compilers_and_libraries_2020/windows/mkl/include` (change it accordingly)
   for gpu faiss,you should also run `set CUDA_PATH = "E:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"` (change it accordingly if you have different version of CUDA at a different location)
1. assume that you have already followed above mentioned approach to build faiss, and all cmake generated project files are located in $  (faiss_root_dir)/build directory
    1. for cpu faiss
    run `cl  "swigfaiss_wrap.cxx"  "../build/lib/Release/faiss.lib" %PythonLibName% "mkl_intel_lp64.lib" "mkl_core.lib" "mkl_intel_thread.lib" "libiomp5md.lib" "kernel32.lib" "user32.lib" "gdi32.lib" "winspool.lib" "shell32.lib" "ole32.lib" "oleaut32.lib" "uuid.lib" "comdlg32.lib" "advapi32.lib" /LD /MD -I%PythonHome%\include -I%PythonHome%\Lib\site-packages\numpy\core\include -I..\..\ /link /MACHINE:X64 /INCREMENTAL:NO /LIBPATH:%MKL_INCLUDE_DIRS%/../lib/intel64 /LIBPATH:%MKL_INCLUDE_DIRS%/../lib/intel64/Release /LIBPATH:%MKL_INCLUDE_DIRS%/../../compiler/lib/intel64 /LIBPATH:%MKL_INCLUDE_DIRS%/../../compiler/lib/intel64/Release /LIBPath:%PythonHome%\libs /OUT:_swigfaiss.pyd`
    1. for gpu faiss
    run `cl  "swigfaiss_wrap.cxx"  "../build/lib/Release/gpufaiss.lib" "../build/lib/Release/faiss.lib" %PythonLibName% "cudart_static.lib" "cublas.lib" "mkl_intel_lp64.lib" "mkl_core.lib" "mkl_intel_thread.lib" "libiomp5md.lib" "kernel32.lib" "user32.lib" "gdi32.lib" "winspool.lib" "shell32.lib" "ole32.lib" "oleaut32.lib" "uuid.lib" "comdlg32.lib" "advapi32.lib" /LD /MD -I“%CUDA_PATH%/include” -I%PythonHome%\include -I%PythonHome%\Lib\site-packages\numpy\core\include -I..\..\ /link /MACHINE:X64 /INCREMENTAL:NO /LIBPATH:%MKL_INCLUDE_DIRS%/../lib/intel64 /LIBPATH:%MKL_INCLUDE_DIRS%/../lib/intel64/Release /LIBPATH:%MKL_INCLUDE_DIRS%/../../compiler/lib/intel64 /LIBPATH:%MKL_INCLUDE_DIRS%/../../compiler/lib/intel64/Release /LIBPath:"%CUDA_PATH%/lib/x64" /LIBPath:%PythonHome%\libs /OUT:_swigfaiss.pyd`
If everything runs smooth, you will have _swigfaiss.pyd generated
1. run `python setup.py make`
1. copy generated `faiss` folder to destination folder `$(PythonHome)/Lib`, there should be 3 files in this folder, namely __init__.py, _swigfaiss.pyd, swigfaiss.py

### Linux (TBC)