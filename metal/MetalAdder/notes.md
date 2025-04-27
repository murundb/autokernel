Based on https://github.com/larsgeb/m1-gpu-cpp

My Mac has Applle M1 with 8 GPU cores

```
xcrun -sdk macosx metal -c MetalAdder/add.metal -o MetalAdder/add.air
xcrun -sdk macosx metallib MetalAdder/add.air -o MetalAdder/default.metallib
```

```
/opt/homebrew/opt/llvm/bin/clang++ -std=c++17 -stdlib=libc++ -O2 -L/opt/homebrew/opt/libomp/lib -fopenmp -I./metal-cpp -fno-objc-arc -framework Metal -framework Foundation -framework MetalKit -g MetalAdder/main.cpp MetalAdder/MetalAdder.cpp -o 01-MetalAdder/benchmark.x
```