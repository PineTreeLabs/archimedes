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
	-c "program build/Release/imu_filter.elf verify reset exit"
```

### See output in terminal

```bash
screen /dev/tty.usbmodem142103 115200
```

Exit `screen` with `Ctrl + A , Ctrl + \`


## Profiling

Note I2C read takes ~150 µs

By-hand implementation: 9 µs


## Visualization

Requires:

```bash
uv pip install pyqt5 vispy
```