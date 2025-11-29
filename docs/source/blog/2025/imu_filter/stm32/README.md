# Sensor Fusion

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
	-c "program build/Release/spi_chat_controller.elf verify reset exit"
```

### See output in terminal

```bash
screen /dev/tty.usbmodem142103 115200
```

Exit `screen` with `Ctrl + A , Ctrl + \`


## Visualization

Requires:

```bash
uv pip install pyqt5 vispy
```