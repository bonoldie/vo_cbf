#!/usr/bin/env python3
"""CRSF Cockpit — Python / GTK 3 desktop controller."""
import argparse
import math
import time
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib
from crsf import initial_channels
from link import Link

class Stick(Gtk.DrawingArea):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.x = self.y = 0
        self.dragging = False
        self.set_size_request(310, 270)
        self.set_hexpand(True)
        self.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.POINTER_MOTION_MASK)
        self.connect('draw', self.draw)
        self.connect('button-press-event', self.press)
        self.connect('motion-notify-event', self.motion)
        self.connect('button-release-event', self.release)
        self.connect('grab-broken-event', self.release)

    def geometry(self):
        return self.get_allocated_width()/2, self.get_allocated_height()/2, min(self.get_allocated_width(),self.get_allocated_height())*.36

    def update(self,e):
        cx,cy,r = self.geometry()
        x,y = (e.x-cx)/r, (cy-e.y)/r
        length = max(1,math.hypot(x,y))
        self.x,self.y = x/length,y/length
        self.callback(self.x,self.y)
        self.queue_draw()

    def press(self,w,e):
        if e.button == 1:
            self.dragging = True
            self.grab_add()
            self.update(e)
        return True

    def motion(self,w,e):
        if self.dragging:
            self.update(e)
        return True

    def release(self,*args):
        if Gtk.grab_get_current() == self:
            self.grab_remove()
        self.dragging = False
        self.x = self.y = 0
        self.callback(0,0)
        self.queue_draw()
        return True

    def draw(self,w,c):
        cx,cy,r = self.geometry()
        c.set_source_rgb(.06,.09,.14)
        c.paint()
        c.set_line_width(1)
        c.set_source_rgb(.24,.32,.43)
        for size in (.5,1):
            c.arc(cx,cy,r*size,0,2*math.pi)
            c.stroke()
        c.move_to(cx-r,cy); c.line_to(cx+r,cy)
        c.move_to(cx,cy-r); c.line_to(cx,cy+r); c.stroke()
        c.set_source_rgb(.2,.8,.8)
        c.set_line_width(3)
        c.move_to(cx,cy); c.line_to(cx+self.x*r,cy-self.y*r); c.stroke()
        c.arc(cx+self.x*r,cy-self.y*r,13,0,2*math.pi); c.fill()
        c.set_source_rgb(.8,.86,.92)
        c.set_font_size(12)
        c.move_to(cx-44,18); c.show_text('FORWARD')
        c.move_to(12,cy-8); c.show_text('LEFT')
        c.move_to(self.get_allocated_width()-48,cy-8); c.show_text('RIGHT')
        return False

class Cockpit(Gtk.Window):
    def __init__(self,args):
        super().__init__(title='CRSF Cockpit')
        self.set_default_size(1040,820)
        self.args = args
        self.link = None
        self.armed = False
        self.busy = False
        self.channels = initial_channels()
        self.connect('delete-event',self.close)
        self.connect('focus-out-event',self.focus_lost)
        self.connect('key-press-event',self.key)
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=10,margin=16)
        scroll = Gtk.ScrolledWindow()
        scroll.add(root); self.add(scroll)
        title = Gtk.Label(xalign=0)
        title.set_markup('<span size="xx-large" weight="bold">CRSF Cockpit</span>   •   AETR / 16 channels')
        root.pack_start(title,False,False,0)
        root.pack_start(Gtk.Label(label='Angle steering • body-relative tilt commands • altitude hold runs on your flight controller',xalign=0),False,False,0)
        row = Gtk.Box(spacing=8); root.pack_start(row,False,False,0)
        self.port = Gtk.Entry(text=args.port); self.port.set_hexpand(True)
        row.pack_start(self.port,True,True,0)
        self.baud = Gtk.SpinButton.new_with_range(9600,2000000,100)
        self.baud.set_value(args.baud); row.pack_start(self.baud,False,False,0)
        self.demo = Gtk.CheckButton(label='Demo'); self.demo.set_active(args.demo)
        row.pack_start(self.demo,False,False,0)
        self.connect_button = Gtk.Button(label='Connect')
        self.connect_button.connect('clicked',self.connection)
        row.pack_start(self.connect_button,False,False,0)
        self.status = Gtk.Label(label='DISCONNECTED',xalign=0)
        root.pack_start(self.status,False,False,0)
        self.notice = Gtk.Label(label='Test with props removed. Confirm AUX ranges, throttle behaviour and FC failsafe before flight.',xalign=0,wrap=True)
        root.pack_start(self.notice,False,False,0)
        columns = Gtk.Box(spacing=18); root.pack_start(columns,True,True,0)
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=8)
        right = Gtk.Box(orientation=Gtk.Orientation.VERTICAL,spacing=8)
        columns.pack_start(left,True,True,0); columns.pack_start(right,True,True,0)
        self.stick = Stick(self.steer); left.pack_start(self.stick,True,True,0)
        self.stick_readout = Gtk.Label(label='Roll 1500 µs   Pitch 1500 µs')
        left.pack_start(self.stick_readout,False,False,0)
        self.authority = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL,25,500,25)
        self.authority.set_value(150)
        left.pack_start(Gtk.Label(label='Maximum stick deflection (µs from centre)',xalign=0),False,False,0)
        left.pack_start(self.authority,False,False,0)
        self.invert = Gtk.CheckButton(label='Invert pitch direction (verify in Receiver tab)')
        left.pack_start(self.invert,False,False,0)
        self.yaw = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL,1000,2000,1)
        self.yaw.set_value(1500)
        self.yaw.connect('value-changed',lambda s:self.set_channel(3,s.get_value()))
        self.yaw.connect('button-release-event',lambda *a:self.centre_yaw())
        left.pack_start(Gtk.Label(label='Yaw • release mouse to centre',xalign=0),False,False,0)
        left.pack_start(self.yaw,False,False,0)
        self.throttle = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL,1000,2000,1)
        self.throttle.set_value(1000)
        self.throttle.connect('value-changed',lambda s:self.set_channel(2,s.get_value()))
        left.pack_start(Gtk.Label(label='Throttle / FC altitude-control input (µs)',xalign=0),False,False,0)
        left.pack_start(self.throttle,False,False,0)
        neutral = Gtk.Button(label='Set throttle to 1500 µs')
        neutral.connect('clicked',lambda b:self.throttle.set_value(1500))
        left.pack_start(neutral,False,False,0)
        self.telemetry_labels = {}
        for key in ('mode','battery','link','attitude','gps','baro','vario'):
            box = Gtk.Frame(label=key.upper())
            label = Gtk.Label(label='Waiting for telemetry…',xalign=0,wrap=True,selectable=True,margin=10)
            box.add(label); right.pack_start(box,True,True,0)
            self.telemetry_labels[key] = label
        controls = Gtk.Grid(column_spacing=16,row_spacing=8)
        root.pack_start(controls,False,False,0)
        self.mode = self.combo(['Angle (1000)','Neither / Acro (1500)','Horizon (2000)'],0,5)
        self.hold = self.combo(['AltHold (1000)','Neither (1500)','PosHold (2000)'],1,8)
        controls.attach(Gtk.Label(label='AUX2 • attitude',xalign=0),0,0,1,1)
        controls.attach(self.mode,1,0,1,1)
        controls.attach(Gtk.Label(label='AUX5 • hold',xalign=0),2,0,1,1)
        controls.attach(self.hold,3,0,1,1)
        self.toggles = {}
        for n,(name,index) in enumerate([('Mag • AUX3',6),('Failsafe • AUX4',7),('GPSRescue • AUX6',9),('AutoPilot • AUX7',10),('Beeper • AUX8',11),('VTX Pit • AUX9',12)]):
            check = Gtk.CheckButton(label=name)
            check.connect('toggled',lambda w,i=index:self.set_channel(i,2000 if w.get_active() else 1000))
            controls.attach(check,n%3,(n//3)+1,1,1)
            self.toggles[index] = check
        bottom = Gtk.Box(spacing=8); root.pack_start(bottom,False,False,0)
        self.ready = Gtk.CheckButton(label='Receiver/AUX/failsafe verified')
        bottom.pack_start(self.ready,True,True,0)
        self.arm_button = Gtk.Button(label='Request ARM')
        self.arm_button.connect('clicked',self.arm)
        bottom.pack_start(self.arm_button,False,False,0)
        disarm = Gtk.Button(label='DISARM (motors stop)')
        disarm.connect('clicked',self.disarm)
        bottom.pack_start(disarm,False,False,0)
        failsafe = Gtk.Button(label='Stage-2 FAILSAFE')
        failsafe.connect('clicked',lambda b:self.toggles[7].set_active(True))
        bottom.pack_start(failsafe,False,False,0)
        self.channel_label = Gtk.Label(xalign=0,selectable=True)
        root.pack_start(self.channel_label,False,False,0)
        self.diagnostics = Gtk.Label(xalign=0,wrap=True,selectable=True)
        root.pack_start(self.diagnostics,False,False,0)
        GLib.timeout_add(50,self.tick)
        self.show_all()

    def combo(self,labels,active,index):
        widget = Gtk.ComboBoxText()
        for label in labels: widget.append_text(label)
        widget.set_active(active)
        widget.connect('changed',lambda w:self.set_channel(index,[1000,1500,2000][w.get_active()]))
        return widget

    def set_channel(self,index,value):
        self.channels[index] = round(value)

    def steer(self,x,y):
        amount = self.authority.get_value() if hasattr(self,'authority') else 150
        self.channels[0] = round(1500+x*amount)
        self.channels[1] = round(1500+y*amount*(-1 if hasattr(self,'invert') and self.invert.get_active() else 1))

    def centre_yaw(self):
        self.yaw.set_value(1500)
        return False

    def focus_lost(self,*args):
        self.stick.release()
        self.centre_yaw()
        return False

    def key(self,w,event):
        if event.keyval == Gdk.KEY_Escape:
            self.focus_lost()
            self.notice.set_text('Steering centred; throttle and ARM request retained. Escape does not stop motors.')
            return True
        return False

    def connection(self,button):
        if self.busy: return
        if self.link:
            if self.armed:
                self.notice.set_text('Disarm before disconnecting. Connection kept open.')
                return
            self.busy = True
            self.connect_button.set_sensitive(False)
            self.disarm()
            GLib.timeout_add(300,self.finish_disconnect)
            return
        self.disarm()
        self.mode.set_active(0); self.hold.set_active(1)
        for toggle in self.toggles.values(): toggle.set_active(False)
        self.ready.set_active(False)
        self.link = Link(self.port.get_text(),self.baud.get_value_as_int(),self.args.rate,self.demo.get_active(),self.args.rssi_encoding)
        self.link.start()
        self.connect_button.set_label('Disconnect')
        for widget in (self.port,self.baud,self.demo): widget.set_sensitive(False)

    def finish_disconnect(self):
        if self.link:
            self.link.stop()
            if self.link.thread.is_alive(): return True
        self.link = None
        self.busy = False
        self.connect_button.set_label('Connect'); self.connect_button.set_sensitive(True)
        for widget in (self.port,self.baud,self.demo): widget.set_sensitive(True)
        return False

    def arm(self,button):
        if not self.link or self.busy: return
        s = self.link.snapshot(); now = time.monotonic(); t = s['telemetry']
        link = t.get('link',(0,{}))
        fc_fresh = any(k in t and now-t[k][0] < 3 for k in ('mode','battery','attitude','gps','baro'))
        centred = all(abs(self.channels[i]-1500) <= 10 for i in (0,1,3))
        if not (s['opened'] and not s['latched'] and self.ready.get_active() and self.channels[2] <= 1050 and centred and now-link[0]<3 and link[1].get('lq',0)>0 and fc_fresh and all(self.channels[i]<1700 for i in (7,9,10))):
            self.notice.set_text('ARM blocked: verify setup, throttle ≤1050, centred sticks, fresh FC/link telemetry, LQ >0, and failsafe/rescue/autopilot off.')
            return
        self.armed = True; self.channels[4] = 2000
        self.notice.set_text('ARM requested on AUX1. Actual FC arming is not confirmed by the RC command.')

    def disarm(self,*args):
        self.armed = False
        self.channels[4] = 1000
        self.stick.release(); self.centre_yaw(); self.throttle.set_value(1000)
        if self.link: self.link.command(self.channels)
        self.notice.set_text('DISARM requested: AUX1 low, throttle low. Confirm motors stopped.')

    def close(self,*args):
        if self.armed:
            self.notice.set_text('Window kept open: disarm before exiting.')
            return True
        if self.link:
            self.connection(None)
            return True
        Gtk.main_quit()
        return False

    def tick(self):
        self.steer(self.stick.x,self.stick.y)
        self.stick_readout.set_text(f'Roll {self.channels[0]} µs   Pitch {self.channels[1]} µs')
        self.arm_button.set_sensitive(not self.armed and not self.busy)
        self.channel_label.set_text('REQUESTED CHANNELS (µs)\n'+'  '.join(f'{i+1}:{v}' for i,v in enumerate(self.channels)))
        if not self.link:
            self.status.set_text('DISCONNECTED • ARM request LOW')
            for label in self.telemetry_labels.values(): label.set_text('No active connection')
            return True
        s = self.link.snapshot()
        self.link.command(self.channels)
        now = time.monotonic(); t = s['telemetry']
        fresh_link = 'link' in t and now-t['link'][0]<3 and t['link'][1]['lq']>0
        fresh_fc = any(k in t and now-t[k][0]<3 for k in ('mode','battery','attitude','gps','baro'))
        prefix = 'DEMO • ' if self.link.demo else ''
        self.status.set_text(prefix+f"Serial {'OPEN' if s['opened'] else 'CLOSED'}  |  RF {'LIVE' if fresh_link else 'MISSING / STALE'}  |  FC telemetry {'LIVE' if fresh_fc else 'MISSING / STALE'}  |  ARM request {'HIGH' if self.armed else 'LOW'}")
        if s['latched']:
            self.notice.set_text('WATCHDOG LATCHED: worker is centring steering and requesting AUX4 failsafe. Disarm to clear.')
        if s['error']: self.notice.set_text(s['error']+' — serial transmission stopped; receiver/FC failsafe must handle link loss.')
        for name,label in self.telemetry_labels.items():
            if name not in t:
                label.set_text('Waiting for telemetry…'); continue
            timestamp,v = t[name]; age = now-timestamp
            text = self.format_telemetry(name,v)
            label.set_text(text+f"\n{'STALE • ' if age>3 else ''}{age:.1f}s ago")
        counts = '  '.join(f'{a:02X}/{k:02X}:{v}' for (a,k),v in sorted(s['counts'].items()))
        self.diagnostics.set_text(f"TX {s['hz']:.0f} Hz • {s['tx']} frames   RX {s['rx']} bytes   CRC errors {s['bad_crc']}   junk {s['junk']}\nFrames address/type: {counts or 'none'}")
        return True

    @staticmethod
    def format_telemetry(name,v):
        if name == 'mode': return v['name']+' (reported by FC)'
        if name == 'battery': return f"{v['voltage']:.1f} V   {v['current']:.1f} A   {v['used']} mAh   {v['remaining']}%"
        if name == 'attitude': return f"Pitch {v['pitch']:+.1f}°   Roll {v['roll']:+.1f}°   Yaw {v['yaw']:.1f}°"
        if name == 'gps': return f"{v['lat']:.6f}, {v['lon']:.6f}\nGPS altitude {v['altitude']} m • {v['satellites']} satellites\n{v['speed']:.1f} km/h • course {v['heading']:.1f}°"
        if name == 'link': return f"Uplink LQ {v['lq']}% • RSSI {v['rssi1']}/{v['rssi2']} dBm • SNR {v['snr']} dB\nDownlink LQ {v['down_lq']}% • RSSI {v['down_rssi']} dBm\nRF profile {v['rf']} • TX {v['power_mw']} mW"
        if name == 'baro': return f"Relative altitude {v['altitude']:.7f} m • vertical speed {v['vertical_speed']:.7f} m/s"
        return f"Vertical speed {v['vertical_speed']:.2f} m/s"

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',default='/dev/ttyACM0')
    p.add_argument('--baud',type=int,default=115200)
    p.add_argument('--rate',type=float,default=100)
    p.add_argument('--demo',action='store_true')
    p.add_argument('--rssi-encoding',choices=['auto','magnitude'],default='auto',help='auto preserves your ELRS signed-int8 compatibility; magnitude follows TBS spec')
    args = p.parse_args()
    if args.baud < 9600 or not 10 <= args.rate <= min(250,args.baud*.7/260):
        p.error('Choose 10–250 Hz within 70%% serial occupancy (26-byte RC frames)')
    Cockpit(args)
    Gtk.main()
