import time

try:
    import serial
    from serial import SerialException
except ImportError:
    serial = None
    SerialException = Exception


class ArduinoController:
    def __init__(self, port: str = "COM6", baudrate: int = 9600, min_interval: float = 2.0):
        self.port = port
        self.baudrate = baudrate
        self.min_interval = min_interval

        self.ser = None
        self.last_command: str | None = None
        self.last_sent_time = 0.0

    def connect(self) -> bool:
        if serial is None:
            print("[ARDUINO] pyserial не установлен")
            return False

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2.0)
            print(f"[ARDUINO] Подключено к {self.port}")
            return True
        except Exception as e:
            print(f"[ARDUINO] Не удалось подключиться к {self.port}: {e}")
            self.ser = None
            return False

    def is_connected(self) -> bool:
        ser = self.ser
        return ser is not None and ser.is_open

    def send_command(self, command: str, force: bool = False) -> bool:
        now = time.time()

        if not force:
            if self.last_command == command and (now - self.last_sent_time) < self.min_interval:
                return False

        ser = self.ser
        if ser is None or not ser.is_open:
            return False

        try:
            ser.write((command + "\n").encode("utf-8"))
            self.last_command = command
            self.last_sent_time = now
            print(f"[ARDUINO] -> {command}")
            return True
        except SerialException as e:
            print(f"[ARDUINO] Ошибка порта: {e}")
            return False
        except Exception as e:
            print(f"[ARDUINO] Ошибка отправки: {e}")
            return False

    def access_granted(self) -> bool:
        return self.send_command("ACCESS_GRANTED")

    def access_denied(self) -> bool:
        return self.send_command("ACCESS_DENIED")

    def idle(self, force: bool = False) -> bool:
        return self.send_command("IDLE", force=force)

    def close(self) -> None:
        ser = self.ser
        if ser is None:
            return

        try:
            if ser.is_open:
                ser.close()
                print("[ARDUINO] Порт закрыт")
        except Exception:
            pass
        finally:
            self.ser = None