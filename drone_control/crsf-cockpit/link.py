"""Independent serial scheduler, GUI heartbeat watchdog and thread-safe snapshots."""
import copy
import threading
import time
from crsf import Parser, decode, rc_frame, initial_channels, watchdog_channels

class Link:
    def __init__(self, port, baud=115200, rate=100, demo=False, encoding='auto', serial_factory=None):
        self.port, self.baud, self.rate, self.demo = port, baud, rate, demo
        self.encoding, self.serial_factory = encoding, serial_factory
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.channels = initial_channels()
        self.heartbeat = time.monotonic()
        self.telemetry = {}
        self.parser = Parser()
        self.opened = False
        self.error = ''
        self.latched = False
        self.tx_count = self.rx_bytes = 0
        self.period = max(1 / rate, 1 / 250, 260 / (baud * .7))
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def command(self, channels):
        rc_frame(channels)  # Validate before accepting.
        with self.lock:
            self.channels = list(channels)
            self.heartbeat = time.monotonic()
            # Only an explicit disarm clears a watchdog trip.
            if channels[4] < 1700:
                self.latched = False

    def snapshot(self):
        with self.lock:
            return dict(opened=self.opened, error=self.error, latched=self.latched,
                        telemetry=copy.deepcopy(self.telemetry), tx=self.tx_count,
                        rx=self.rx_bytes, bad_crc=self.parser.bad_crc,
                        junk=self.parser.junk, counts=dict(self.parser.counts), hz=1/self.period)

    def stop(self):
        self.stop_event.set()

    def run(self):
        ser = None
        try:
            if not self.demo:
                factory = self.serial_factory
                if factory is None:
                    import serial
                    factory = serial.Serial
                ser = factory(self.port, self.baud, timeout=0, write_timeout=.05, exclusive=True)
            with self.lock:
                self.opened = True
            next_tick = time.monotonic()
            next_demo = 0
            while not self.stop_event.is_set():
                now = time.monotonic()
                with self.lock:
                    if now - self.heartbeat > .5 and self.channels[4] > 1700:
                        self.latched = True
                    out = watchdog_channels(self.channels) if self.latched else list(self.channels)
                if now >= next_tick:
                    packet = rc_frame(out)
                    if ser and ser.write(packet) != len(packet):
                        raise IOError('Partial serial write; transmission stopped')
                    with self.lock:
                        self.tx_count += 1
                    next_tick = now + self.period
                if ser:
                    data = ser.read(min(ser.in_waiting, 4096))
                    with self.lock:
                        self.rx_bytes += len(data)
                        for addr, kind, payload in self.parser.feed(data, now):
                            item = decode(kind,payload,self.encoding)
                            if item:
                                name, value = item
                                self.telemetry[name] = (now,value)
                                if name == 'timing':
                                    # UART occupancy <= 70%, cap 250 Hz. Offset is diagnostic only.
                                    self.period = max(value['period'], 1/250, 260/(self.baud*.7))
                elif now >= next_demo:
                    import math
                    values = {
                        'battery':dict(voltage=16.2,current=2.3,used=240,remaining=82),
                        'attitude':dict(pitch=8*math.sin(now),roll=12*math.cos(now*.5),yaw=42),
                        'gps':dict(lat=45.4384,lon=10.9916,speed=0,heading=42,altitude=83,satellites=14),
                        'baro':dict(altitude=2.5,vertical_speed=0),
                        'mode':dict(name='DEMO — no FC'),
                        'link':dict(rssi1=-65,rssi2=-68,lq=99,snr=9,antenna=0,rf=5,power_mw=100,down_rssi=-70,down_lq=98,down_snr=7)}
                    with self.lock:
                        self.telemetry.update({k:(now,v) for k,v in values.items()})
                    next_demo = now+.1
                self.stop_event.wait(min(.003, max(.0005,next_tick-time.monotonic())))
        except Exception as exc:
            with self.lock:
                self.error = f'{type(exc).__name__}: {exc}'
        finally:
            if ser:
                ser.close()
            with self.lock:
                self.opened = False
