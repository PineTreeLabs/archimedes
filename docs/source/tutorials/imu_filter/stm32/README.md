# Sensor Fusion

LinkedIn post:

```
Think in Python, deploy in C.


```

## Deploy

### Compiler optimizations:

In cmake/gcc-arm-none-eabi.cmake:

```
set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_RELEASE "-O3 -g0")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g0")
```

### Flash firmware

```bash
cmake --preset=Release
cmake --build --preset=Release
openocd -f interface/stlink.cfg -f target/stm32h7x.cfg \
	-c "program build/Release/f16_hil.elf verify reset exit"
```