import struct
import time
import unittest
from crsf import *
from link import Link

class CodecTests(unittest.TestCase):
    def test_crc_known_vector(self):
        self.assertEqual(crc8(b'123456789'),0xBC)

    def test_rc_channels_independent_unpack(self):
        channels = initial_channels()
        channels[0],channels[1],channels[15] = 2000,1000,1750
        packet = rc_frame(channels)
        self.assertEqual(len(packet),26)
        self.assertEqual(packet[:3],bytes([0xEE,24,0x16]))
        bits = int.from_bytes(packet[3:-1],'little')
        decoded = [(((bits >> (11*i)) & 2047)-992)*5/8+1500 for i in range(16)]
        self.assertEqual(decoded,channels)

    def test_fragmented_noise_echo_crc(self):
        parser = Parser()
        good = frame(8,struct.pack('>HH',160,23)+bytes([0,1,44,80]),0xEA)
        bad = bytearray(good); bad[-1] ^= 1
        result = []
        for byte in b'noise'+bad+rc_frame(initial_channels())+good:
            result.extend(parser.feed(bytes([byte])))
        self.assertEqual([x[1] for x in result],[0x16,8])
        self.assertGreater(parser.bad_crc,0)
        self.assertEqual(decode(8,result[-1][2])[1]['used'],300)

    def test_partial_header_timeout_recovers(self):
        p = Parser()
        good = frame(0x21,b'ANGLE\0',0xEA)
        self.assertEqual(p.feed(bytes([0xC8,62])+good,0),[])
        self.assertEqual(p.feed(b'',.2)[0][1],0x21)

    def test_telemetry(self):
        topic,gps = decode(2,struct.pack('>iiHHHB',454000000,-109000000,123,12345,1083,12))
        self.assertEqual((gps['lat'],gps['lon'],gps['altitude']),(45.4,-10.9,83))
        self.assertEqual(decode(9,struct.pack('>Hb',10025,-20))[1]['altitude'],2.5)
        self.assertAlmostEqual(decode(9,struct.pack('>Hb',10025,-20))[1]['vertical_speed'],-math.expm1(.52))
        self.assertIsNone(decode(9,b'\xff\xff')[1]['altitude'])
        for kind in (2,7,8,9,0x14,0x1E,0x21,0x3A):
            self.assertIsNone(decode(kind,b''))
        self.assertEqual(rssi(190,'auto'),-66)
        self.assertEqual(rssi(190,'magnitude'),-190)

    def test_timing_destination_and_scale(self):
        payload = bytes([0xEA,0xEE,0x10])+struct.pack('>ii',100000,-100)
        self.assertEqual(decode(0x3A,payload)[1]['period'],.01)
        self.assertIsNone(decode(0x3A,b'\x00'+payload[1:]))

    def test_startup_aux_and_watchdog(self):
        c = initial_channels()
        self.assertEqual(c[4:13],[1000,1000,1000,1000,1500,1000,1000,1000,1000])
        c[4]=2000; c[2]=1480; c[0]=1700
        safe = watchdog_channels(c)
        self.assertEqual((safe[0],safe[2],safe[4],safe[7]),(1500,1480,2000,2000))

class WorkerTests(unittest.TestCase):
    def test_serial_echo_and_watchdog_latch(self):
        class Fake:
            def __init__(self,*a,**k): self.data=bytearray(); self.packets=[]; self.closed=False
            @property
            def in_waiting(self): return len(self.data)
            def read(self,n):
                out=bytes(self.data[:n]); del self.data[:n]; return out
            def write(self,p): self.packets.append(p); self.data.extend(p); return len(p)
            def close(self): self.closed=True
        fake=Fake()
        link=Link('test',serial_factory=lambda *a,**k:fake)
        c=initial_channels(); c[4]=2000; c[2]=1460
        link.command(c); link.start()
        try:
            time.sleep(.65)
            s=link.snapshot()
            self.assertTrue(s['latched'])
            self.assertEqual(s['telemetry'],{}) # Echo cannot qualify as a drone.
            bits=int.from_bytes(fake.packets[-1][3:-1],'little')
            self.assertEqual(((bits>>(7*11))&2047),1792)
            link.command(c); time.sleep(.03)
            self.assertTrue(link.snapshot()['latched'])
            link.command(initial_channels()); time.sleep(.03)
            self.assertFalse(link.snapshot()['latched'])
        finally:
            link.stop(); link.thread.join(1)
        self.assertTrue(fake.closed)
        self.assertFalse(link.thread.is_alive())

    def test_serial_failure_reported(self):
        def fail(*a,**k): raise OSError('port unavailable')
        link=Link('missing',serial_factory=fail); link.start(); link.thread.join(1)
        self.assertIn('port unavailable',link.snapshot()['error'])
        self.assertFalse(link.snapshot()['opened'])

if __name__ == '__main__': unittest.main()
