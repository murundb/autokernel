My Mac has Applle M1 with 8 GPU cores

To compile the custom kernel benchmark:

```
xcrun -sdk macosx metal -c MetalMatrixMultiplier/matmul.metal -o MetalMatrixMultiplier/matmul.air
xcrun -sdk macosx metallib MetalMatrixMultiplier/matmul.air -o MetalMatrixMultiplier/default.metallib
```

```
/opt/homebrew/opt/llvm/bin/clang++ -std=c++17 -stdlib=libc++ -O2 -L/opt/homebrew/opt/libomp/lib -fopenmp -I./metal-cpp -fno-objc-arc -framework Metal -framework Foundation -framework MetalKit -g MetalMatrixMultiplier/main.cpp MetalMatrixMultiplier/MetalMatrixMultiplier.cpp -o MetalMatrixMultiplier/custom_benchmark.x
```

To compile the native benchmark:

```
/opt/homebrew/opt/llvm/bin/clang++ -std=c++17 -stdlib=libc++ -fobjc-arc -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework QuartzCore MetalMatrixMultiplier/main.mm -o MetalMatrixMultiplier/mps_benchmark.x
```

