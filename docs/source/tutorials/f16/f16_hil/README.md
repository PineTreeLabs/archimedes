# F-16 HIL Testing

1. ✅ Verify USART3 connection.  Section 7.6.5 (Table 10) in UM2407 for MB1364 Nucleo-144 dev boards (STM32H7) suggests this is enabled for Virtual COM (instead of ST-Morpho) by default.

2. ✅ Verify plant runs with no external I/O. Should be able to profile RK4 step time for steady turn (expect ~10 kHz) and get data back over USB.  **Publish LinkedIn post!**

3. Design pitch controller for `xcg=0.4` (Chapter 4 in SLJ)

4. Add sensor models:
	- Accelerometer
	- Rate gyro
	- Pitot-static system
	- AoA/AoS vanes
	- LVDT/RVDT for control surfaces?

5. Design full longitudinal flight control system

6. Simple HIL testing: attitude controller (4 control signals) with sensor feedback (~16 sensors). Target 1 kHz.

**Ship it!**

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