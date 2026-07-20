from pynput.keyboard import Key, Listener, KeyCode

class  Playback:
    """
    Handles scene playback based on keyboard inputs
    """
    
    paused = True
    ctrl = False
    show_obstacles_collision_boxes = True

    vertical_offset = 0.0
    horizontal_offset = 0.0

    step = 0

    def __init__(self):
        listener = Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

    def on_press(self, key):
        if key == Key.space:
            self.paused = not self.paused

        if key == Key.up:
            self.vertical_offset += 1

        if key == Key.down:
            self.vertical_offset -= 1

        if key == Key.right:
            self.horizontal_offset += 1

        if key == Key.left:
            self.horizontal_offset -= 1

        if key == KeyCode.from_char('o'):
            self.show_obstacles_collision_boxes = not self.show_obstacles_collision_boxes


        if key == Key.ctrl:
            self.ctrl = True


            
    def on_release(self, key):
        if key == Key.ctrl:
            self.ctrl = False

        if key == KeyCode.from_char('n') and self.ctrl:
            self.step = self.step + 1
            print(f"[Playback] requested steps: {self.step}")