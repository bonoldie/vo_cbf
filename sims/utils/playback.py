from pynput.keyboard import Key, Listener, KeyCode

class  Playback:
    """
    Handles scene playback based on keyboard inputs
    """
    
    paused = True
    ctrl = False
    show_obstacles_collision_boxes = True

    def __init__(self):
        listener = Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

    def on_press(self, key):
        if key == Key.space:
            self.paused = not self.paused

        if key == KeyCode.from_char('o'):
            self.show_obstacles_collision_boxes = not self.show_obstacles_collision_boxes

        if key == Key.ctrl:
            self.ctrl = True
            
    def on_release(self, key):
        if key == Key.ctrl:
            self.ctrl = False