"""CRSF wire codec; no GTK or serial dependencies."""
import math
import struct
import time
from collections import Counter

ADDRESSES = {0x00, 0x10, 0x12, 0x80, 0x8A, 0xC0, 0xC2, 0xC4, 0xC8, 0xCA, 0xCC, 0xEA, 0xEC, 0xEE, 0xEF}
FC_TYPES = {0x02, 0x07, 0x08, 0x09, 0x1E, 0x21}

def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ (0xD5 if crc & 128 else 0)) & 255
    return crc

def frame(kind, payload, address=0xEE):
    body = bytes([kind]) + bytes(payload)
    if len(body) > 61:
        raise ValueError('Frame exceeds 64 bytes')
    return bytes([address, len(body) + 1]) + body + bytes([crc8(body)])

def rc_frame(channels):
    if len(channels) != 16:
        raise ValueError('Exactly 16 channels required')
    bits = 0
    for i, value in enumerate(channels):
        if not math.isfinite(value) or not 988 <= value <= 2012:
            raise ValueError('Invalid channel value')
        ticks = max(172, min(1811, round((value - 1500) * 8 / 5 + 992)))
        bits |= ticks << (i * 11)
    return frame(0x16, bits.to_bytes(22, 'little'))

class Parser:
    def __init__(self):
        self.buffer = bytearray()
        self.bad_crc = self.junk = 0
        self.counts = Counter()
        self.partial_since = None

    def feed(self, data, now=None):
        now = time.monotonic() if now is None else now
        self.buffer.extend(data)
        result = []
        while len(self.buffer) >= 2:
            b = self.buffer
            if b[0] not in ADDRESSES or not 2 <= b[1] <= 62:
                del b[0]
                self.junk += 1
                self.partial_since = None
                continue
            size = b[1] + 2
            if len(b) < size:
                if self.partial_since is None:
                    self.partial_since = now
                if now - self.partial_since > .1:
                    del b[0]
                    self.junk += 1
                    self.partial_since = None
                    continue
                break
            self.partial_since = None
            if crc8(b[2:size-1]) != b[size-1]:
                del b[0]
                self.bad_crc += 1
                continue
            addr, kind, payload = b[0], b[2], bytes(b[3:size-1])
            del b[:size]
            self.counts[(addr, kind)] += 1
            result.append((addr, kind, payload))
        return result

def rssi(raw, encoding='magnitude'):
    return raw - 256 if encoding == 'auto' and raw >= 128 else -raw

def decode(kind, p, encoding='magnitude'):
    """Return (topic, data), rejecting short frames but accepting extra fields."""
    minimum = {2:15, 7:2, 8:8, 9:2, 0x14:10, 0x1E:6, 0x21:1, 0x3A:11}
    if kind not in minimum or len(p) < minimum[kind]:
        return None
    if kind == 2:
        lat, lon, speed, heading, alt, sats = struct.unpack_from('>iiHHHB', p)
        return 'gps', dict(lat=lat/1e7, lon=lon/1e7, speed=speed/10, heading=heading/100, altitude=alt-1000, satellites=sats)
    if kind == 7:
        return 'vario', {'vertical_speed': struct.unpack_from('>h', p)[0]/100}
    if kind == 8:
        v, a = struct.unpack_from('>HH', p)
        return 'battery', dict(voltage=v/10, current=a/10, used=int.from_bytes(p[4:7], 'big'), remaining=p[7])
    if kind == 9:
        alt = int.from_bytes(p[:2], 'big')
        speed = int.from_bytes(p[2:3], 'big', signed=True) if len(p) >= 3 else None
        return 'baro', dict(altitude=None if alt == 65535 else ((alt & 32767) if alt & 32768 else alt/10-1000), vertical_speed=None if speed is None else math.copysign(math.expm1(abs(speed)*.026), speed))
    if kind == 0x14:
        a,b,lq,snr,ant,rf,power,d,dlq,dsnr = struct.unpack_from('>BBBbBBBBBb', p)
        powers = [0,10,25,100,500,1000,2000,250,50]
        return 'link', dict(rssi1=rssi(a,encoding), rssi2=rssi(b,encoding), lq=lq, snr=snr, antenna=ant, rf=rf, power_mw=powers[power] if power < len(powers) else None, down_rssi=rssi(d,encoding), down_lq=dlq, down_snr=dsnr)
    if kind == 0x1E:
        return 'attitude', dict(zip(('pitch','roll','yaw'), (math.degrees(x/10000) for x in struct.unpack_from('>hhh',p))))
    if kind == 0x21:
        return 'mode', {'name':p.split(b'\0',1)[0].decode('ascii','replace')}
    if kind == 0x3A and p[:3] == bytes([0xEA,0xEE,0x10]):
        rate, offset = struct.unpack_from('>ii',p,3)
        if 5000 <= rate <= 1000000:
            return 'timing', {'period':rate/1e7, 'offset':offset/1e7}
    return None

# Index 0..15 = CH1..CH16. Middle values deliberately disable AUX5 holds.
def initial_channels():
    return [1500,1500,1000,1500,1000,1000,1000,1000,1500,1000,1000,1000,1000,1000,1000,1000]

def watchdog_channels(channels):
    result = list(channels)
    result[0] = result[1] = result[3] = 1500
    if result[4] > 1700:
        result[7] = 2000  # Request the user's configured stage-2 failsafe.
    else:
        result[2] = 1000
    return result
