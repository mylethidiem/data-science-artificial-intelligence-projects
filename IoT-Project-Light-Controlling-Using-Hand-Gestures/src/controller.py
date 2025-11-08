import platform  # Import platform module to get the operating system
import time  # Import time module for time-related functions

import serial  # Import pySerial module for serial communication
from serial.tools import list_ports  # Import list_ports to list available serial ports

# Define Modbus commands for controlling relays
"""
Module supporting hardware device control via Modbus RTU RS485 communication
"""

print("Sensor and Actuator Control Module")

"""
Explanation of Modbus Command Structure
- Device Address: The first byte (e.g., 1) is the device's Modbus address.
- Function Code: The second byte (e.g., 5) specifies the function code.
    In this case, 5 is the function code for writing a single coil (turning a relay on or off).
- Coil Address: The third and fourth bytes (e.g., 0, 0 or 0, 1 or 0, 2)
    specify the address of the coil (relay) to be controlled.
- Data: The fifth and sixth bytes (e.g., 0xFF, 0 or 0, 0) is the data to be written to the coil.
    0xFF, 0 typically means turning the relay on, and 0, 0 means turning it off.
- CRC: The last two bytes (e.g., 0x8C, 0x3A or 0xCD, 0xCA)
    are the Cyclic Redundancy Check (CRC) for error-checking.
"""
RELAY1_ON = [1, 5, 0, 0, 0xFF, 0, 0x8C, 0x3A]  # Command to turn on relay 1
RELAY1_OFF = [1, 5, 0, 0, 0, 0, 0xCD, 0xCA]  # Command to turn off relay 1

RELAY2_ON = [1, 5, 0, 1, 0xFF, 0, 0xDD, 0xFA]  # Command to turn on relay 2
RELAY2_OFF = [1, 5, 0, 1, 0, 0, 0x9C, 0x0A]  # Command to turn off relay 2

RELAY3_ON = [1, 5, 0, 2, 0xFF, 0, 0x2D, 0xFA]  # Command to turn on relay 3
RELAY3_OFF = [1, 5, 0, 2, 0, 0, 0x6C, 0x0A]  # Command to turn off relay 3


class ModbusMaster:
    # Class for controlling actuators using Modbus protocol
    def __init__(self) -> None:
        port_list = list_ports.comports()  # List available serial ports
        print(f"Available ports: {port_list}")
        if len(port_list) == 0:
            raise Exception("No port found!")

        which_os = platform.system()  # Get the operating system
        if which_os == "Linux":
            name_ports = list(
                filter(lambda name: "USB" in name, (port.name for port in port_list))
            )
            portName = "/dev/" + name_ports[0]  # Get the first USB port
            print(portName)  # Print the port name
        else:
            portName = "None"
            for port in port_list:
                strPort = str(port)
                if "USB Serial" in strPort:
                    splitPort = strPort.split(" ")
                    portName = splitPort[0]  # Get the port name for Windows
        self.ser = serial.Serial(portName)  # Open the serial port
        self.ser.is_open  # The serial port is open
        self.ser.baudrate = 9600
        self.ser.stopbits = serial.STOPBITS_ONE
        self.ser.parity = serial.PARITY_NONE
        self.ser.bytesize = serial.EIGHTBITS
        print(self.ser.baudrate, self.ser.stopbits, self.ser.parity, self.ser.bytesize)

    def __enter__(self):
        return self

    def __exit__(self):
        print("closing the serial connection")
        self.close()

    def switch_actuator_1(self, state):
        # Switch actuator 1 on or off
        if state is True:
            self.ser.write(RELAY1_ON)
        else:
            self.ser.write(RELAY1_OFF)

    def switch_actuator_2(self, state):
        # Switch actuator 2 on or off
        if state is True:
            self.ser.write(RELAY2_ON)
        else:
            self.ser.write(RELAY2_OFF)

    def switch_actuator_3(self, state):
        # Switch actuator 3 on or off
        if state is True:
            self.ser.write(RELAY3_ON)
        else:
            self.ser.write(RELAY3_OFF)

    def close(self):
        self.ser.close()  # Close the serial port

    def serial_read_data(self):
        # Read data from the serial port
        ser = self.ser
        bytesToRead = (
            ser.inWaiting()
        )  # Get the number of bytes waiting in the serial buffer
        if bytesToRead > 0:
            print("Data received")
            data = ser.read(bytesToRead)  # Read the bytes from the serial buffer
            data_array = list(data)  # Convert the bytes to a list
            print(data_array)
            if len(data_array) >= 7:
                array_size = len(data_array)
                value = (
                    data_array[array_size - 4] * 256 + data_array[array_size - 3]
                )  # Calculate the value from the data array
                return value
            else:
                return -1  # Return -1 if the data array is less than 7
        return 0  # Return 0 if no data is read


if __name__ == "__main__":
    a = ModbusMaster()
    while True:
        print("run")
        a.switch_actuator_1(True)
        # time.sleep(0.03)
        a.switch_actuator_3(True)
        time.sleep(2)  # Wait for 2 seconds
        a.switch_actuator_1(False)
        time.sleep(0.03)
        a.switch_actuator_3(False)
        time.sleep(1)
