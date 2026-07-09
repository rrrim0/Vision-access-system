import time
from app.arduino_controller import ArduinoController

arduino = ArduinoController(port="COM6", baudrate=9600)

if arduino.connect():
    time.sleep(1)

    print("Тест: ACCESS_GRANTED")
    arduino.access_granted()
    time.sleep(3)

    print("Тест: ACCESS_DENIED")
    arduino.access_denied()
    time.sleep(3)

    print("Тест: IDLE")
    arduino.idle(force=True)

    arduino.close()
else:
    print("Подключение не удалось")