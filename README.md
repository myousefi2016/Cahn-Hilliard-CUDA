# Cahn Hilliard CUDA (Phase-Field Simulation of Spinodal Decomposition)

Description: This CUDA program does phase-field simulation of spinodal decomposition based on this references:

1. [Cahn-Hilliard Equation](https://en.wikipedia.org/wiki/Cahn%E2%80%93Hilliard_equation)

Dependencies: CUDA Toolkit 9.2, GCC 6.3

In order to compile the program use these commands in UNIX shell terminals:

```
git clone git@github.com:myousefi2016/Cahn-Hilliard-CUDA.git
cd Cahn-Hilliard-CUDA && ./build.sh
```

To run the program after compilation just running this command:

```
mkdir out && ./main
```

It will store the results in ```out``` directory as vtk files. Good luck and if you use this piece of code for your research don't forget to give attribute to this github repository. Also this program is tested on a cluster with Tesla V100 gpus. If you don't have access to this type of gpu, you could reduce the simulation size by changing these lines in ```main.cu``` file.

```
#define DATAXSIZE 128
#define DATAYSIZE 128
#define DATAZSIZE 128
```

![alt text](https://raw.githubusercontent.com/myousefi2016/Cahn-Hilliard-CUDA/master/animation/output.gif)
