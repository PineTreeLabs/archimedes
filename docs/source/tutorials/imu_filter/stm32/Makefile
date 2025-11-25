# st-info --probe to get the serial number of the connected ST-Link
STM32_SN=002400363234510637333934

clean:
	@cmake --build --preset=Release --target clean

build:
	@cmake --preset=Release
	@cmake --build --preset=Release

flash:
	@cmake --build --preset=Release
	@openocd -f interface/stlink.cfg -f target/stm32h7x.cfg -c \
		"adapter serial $(STM32_SN)" -c "program build/Release/imu_fusion.elf verify reset exit"