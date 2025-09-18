# main.py
"""
Focus Face - Kivy Edition
A mobile-ready version of the facial gesture reaction game.
"""
import os
import sys

# --- Definitive Message Suppression ---
# This block must be at the very top, before any other imports.
class DevNull:
    def write(self, msg):
        pass
sys.stderr = DevNull()
sys.stdout = DevNull()
# --- End Suppression ---

import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
import numpy as np
import threading
import time
import random

# Now that messages are suppressed, we can import Kivy
os.environ['KIVY_NO_CONSOLELOG'] = '1'
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.audio import SoundLoader
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.uix.widget import Widget
from kivy.animation import Animation

# Restore standard output after imports are done
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

import config

class GestureDetector:
    """A modified version of the gesture detector to run in a separate thread."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.current_gesture = 'NONE'
        self.running = False
        self.frame = None
        self.lock = threading.Lock()
        
        self.MOUTH_INDICES = {"vertical": [13, 14], "horizontal": [78, 308]}
        self.EYEBROW_INDICES = {"left_top": 105, "left_bottom": 159}
        self.EYE_CORNERS = {"left": 33, "right": 263}

    def start(self):
        self.running = True
        threading.Thread(target=self._detect_loop, daemon=True).start()

    def stop(self):
        self.running = False
        try:
            if self.face_mesh:
                self.face_mesh.close()
        except ValueError:
            pass

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame

    def get_gesture(self):
        return self.current_gesture

    def _calculate_mar(self, landmarks):
        vertical = dist.euclidean(landmarks[0], landmarks[1])
        horizontal = dist.euclidean(landmarks[2], landmarks[3])
        return vertical / horizontal if horizontal > 0 else 0

    def _detect_loop(self):
        while self.running:
            with self.lock:
                frame_copy = self.frame.copy() if self.frame is not None else None
            
            if frame_copy is None:
                time.sleep(0.01)
                continue

            image = cv2.cvtColor(cv2.flip(frame_copy, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.face_mesh.process(image)
            image.flags.writeable = True

            gesture = 'NONE'
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                
                eye_left = face_landmarks[self.EYE_CORNERS["left"]]
                eye_right = face_landmarks[self.EYE_CORNERS["right"]]
                eye_span = dist.euclidean((eye_left.x, eye_left.y), (eye_right.x, eye_right.y))

                mouth_v = [face_landmarks[i] for i in self.MOUTH_INDICES["vertical"]]
                mouth_h = [face_landmarks[i] for i in self.MOUTH_INDICES["horizontal"]]
                mar_coords = [(p.x, p.y) for p in mouth_v] + [(p.x, p.y) for p in mouth_h]
                mar = self._calculate_mar(mar_coords)

                eyebrow_top = face_landmarks[self.EYEBROW_INDICES["left_top"]]
                eyebrow_bottom = face_landmarks[self.EYEBROW_INDICES["left_bottom"]]
                eyebrow_dist = dist.euclidean((eyebrow_top.x, eyebrow_top.y), (eyebrow_bottom.x, eyebrow_bottom.y))
                norm_eyebrow_dist = eyebrow_dist / eye_span
                
                is_mouth_open = mar > config.MOUTH_AR_THRESHOLD
                is_eyebrows_up = norm_eyebrow_dist > config.EYEBROW_RAISE_THRESHOLD

                if is_mouth_open:
                    gesture = 'MOUTH_OPEN'
                elif is_eyebrows_up:
                    gesture = 'EYEBROWS_UP'
            
            self.current_gesture = gesture
            time.sleep(0.01)

class GameWidget(FloatLayout):
    def __init__(self, is_first_launch=True, **kwargs):
        super().__init__(**kwargs)
        self.stars = []

        with self.canvas.before:
            Color(0.05, 0.05, 0.1, 1)
            self.bg_rect = Rectangle(size=self.size, pos=self.pos)
            self._generate_stars()
        
        self.bind(size=self._update_graphics, pos=self._update_graphics)

        self.game_running = False
        self.game_over = False
        
        self.camera = None
        self.detector = None
        
        self.sound_correct = SoundLoader.load(config.SOUND_CORRECT)
        self.sound_miss = SoundLoader.load(config.SOUND_MISS)
        self.sound_start = SoundLoader.load(config.SOUND_START)
        self.sound_gameover = SoundLoader.load(config.SOUND_GAMEOVER)

        self.build_ui()
        
        if is_first_launch:
            self.run_loading_animation()
        else:
            self.start_game()

        Clock.schedule_interval(self.update, config.REFRESH_RATE)
    
    def _generate_stars(self, count=100):
        with self.canvas.before:
            for _ in range(count):
                star = {'pos': [random.uniform(0, Window.width), random.uniform(0, Window.height)],
                        'size': random.uniform(1, 3),
                        'velocity': random.uniform(10, 25)}
                Color(1, 1, 1, random.uniform(0.3, 1.0))
                star['rect'] = Rectangle(pos=star['pos'], size=(star['size'], star['size']))
                self.stars.append(star)

    def _update_graphics(self, instance, value):
        self.bg_rect.pos = instance.pos
        self.bg_rect.size = instance.size
        self._update_borders(instance, value)

    def _update_borders(self, instance, value):
        char_width = self.top_border.font_size * 0.5 
        if char_width > 0:
            num_chars = int(self.width / char_width) - 4
            num_chars = max(0, num_chars)
            self.top_border.text = "╔" + "═" * num_chars + "╗"
            self.bottom_border.text = "╚" + "═" * num_chars + "╝"

    def build_ui(self):
        self.loading_layout = BoxLayout(orientation='vertical', spacing=15, pos_hint={'center_x': 0.5, 'center_y': 0.5}, size_hint=(0.8, 0.4))
        
        ascii_title = Label(text="<< [color=00ffc8]FOCUS FACE[/color] >>", font_name=config.FONT_NAME, font_size='50sp', markup=True)
        boot_title = Label(text="> Boot Sequence Initialized <", font_name=config.FONT_NAME, font_size='25sp', color=(1,1,1,0.7))
        self.loading_status = Label(text="", font_name=config.FONT_NAME, font_size='25sp')
        self.loading_bar = Label(text="[                    ]", font_name=config.FONT_NAME, font_size='30sp')
        
        self.loading_layout.add_widget(ascii_title)
        self.loading_layout.add_widget(boot_title)
        self.loading_layout.add_widget(Widget(size_hint_y=0.2))
        self.loading_layout.add_widget(self.loading_status)
        self.loading_layout.add_widget(self.loading_bar)
        
        self.game_layout = FloatLayout(opacity=0)
        
        self.play_area = FloatLayout(size_hint=(1, 0.8), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.game_layout.add_widget(self.play_area)
        
        self.sync_zone_widget = BoxLayout(orientation='vertical', size_hint=(None, None), size=('300sp', '100sp'), 
                                           pos_hint={'center_x': 0.5, 'center_y': 0.5}, opacity=0)
        
        sync_zone_text = Label(text="SYNC ZONE", font_name=config.FONT_NAME, font_size='30sp', color=(1, 0, 1, 1), size_hint_y=None, height='40sp')
        
        sync_zone_brackets = BoxLayout(orientation='horizontal')
        line_char = '-' * 15
        sync_zone_brackets.add_widget(Label(text=line_char, font_name=config.FONT_NAME, font_size='40sp', color=(1,1,1,0.5)))
        sync_zone_brackets.add_widget(Label(text="|", font_name=config.FONT_NAME, font_size='40sp', color=(1, 0, 1, 1), size_hint_x=None, width='30sp'))
        sync_zone_brackets.add_widget(Widget())
        sync_zone_brackets.add_widget(Label(text="|", font_name=config.FONT_NAME, font_size='40sp', color=(1, 0, 1, 1), size_hint_x=None, width='30sp'))
        sync_zone_brackets.add_widget(Label(text=line_char, font_name=config.FONT_NAME, font_size='40sp', color=(1,1,1,0.5)))
        
        self.sync_zone_widget.add_widget(sync_zone_text)
        self.sync_zone_widget.add_widget(sync_zone_brackets)
        self.play_area.add_widget(self.sync_zone_widget)

        self.top_border = Label(text="", font_name=config.FONT_NAME, font_size='20sp', pos_hint={'top': 1})
        self.bottom_border = Label(text="", font_name=config.FONT_NAME, font_size='20sp', pos_hint={'y': 0})
        self.game_layout.add_widget(self.top_border)
        self.game_layout.add_widget(self.bottom_border)
        
        self.top_panel = BoxLayout(orientation='horizontal', size_hint=(0.95, None), height='40sp', pos_hint={'center_x': 0.5, 'top': 0.98})
        self.score_label = Label(text="Score: 0", font_name=config.FONT_NAME, font_size='30sp', halign='left', size_hint_x=0.3)
        self.rule_label = Label(text="", font_name=config.FONT_NAME, font_size='25sp', halign='center', size_hint_x=0.4)
        self.lives_label = Label(text="", font_name=config.FONT_NAME, font_size='30sp', halign='right', size_hint_x=0.3)
        self.top_panel.add_widget(self.score_label)
        self.top_panel.add_widget(Widget(size_hint_x=0.05))
        self.top_panel.add_widget(self.rule_label)
        self.top_panel.add_widget(Widget(size_hint_x=0.05))
        self.top_panel.add_widget(self.lives_label)
        
        self.bottom_panel = BoxLayout(orientation='horizontal', size_hint=(0.95, None), height='30sp', pos_hint={'center_x': 0.5, 'y': 0.02})
        self.zen_label = Label(text="Zen: [          ]", font_name=config.FONT_NAME, font_size='25sp', halign='left', size_hint_x=0.5)
        self.combo_label = Label(text="Combo: 0x", font_name=config.FONT_NAME, font_size='25sp', halign='right', size_hint_x=0.5)
        self.bottom_panel.add_widget(self.zen_label)
        self.bottom_panel.add_widget(self.combo_label)

        self.game_layout.add_widget(self.top_panel)
        self.game_layout.add_widget(self.bottom_panel)

        self.instruction_panel = Label(text="", font_name=config.FONT_NAME, font_size='35sp', halign='center', valign='middle')
        self.game_layout.add_widget(self.instruction_panel)

        self.game_over_layout = BoxLayout(orientation='vertical', size_hint=(None, None), size=(Window.width * 0.7, Window.height * 0.5),
                                          pos_hint={'center_x': 0.5, 'center_y': 0.5}, spacing=20)
        self.game_over_label = Label(text="Game Over", font_name=config.FONT_NAME, font_size='60sp')
        self.final_score_label = Label(text="Final Score: 0", font_name=config.FONT_NAME, font_size='40sp')
        button_layout = BoxLayout(size_hint_y=None, height='50sp', spacing=20)
        self.restart_button = Button(text="Restart", font_name=config.FONT_NAME, font_size='30sp')
        self.exit_button = Button(text="Exit", font_name=config.FONT_NAME, font_size='30sp')
        self.restart_button.bind(on_press=self.restart_game)
        self.exit_button.bind(on_press=self.exit_game)
        button_layout.add_widget(self.restart_button)
        button_layout.add_widget(self.exit_button)
        self.game_over_layout.add_widget(self.game_over_label)
        self.game_over_layout.add_widget(self.final_score_label)
        self.game_over_layout.add_widget(button_layout)

    def run_loading_animation(self, *args):
        self.add_widget(self.loading_layout)
        self.animation_time = 0
        self.animation_phase = 0
        self.typing_index = 0
        self.status_messages = [
            "Calibrating biometric sensors...",
            "Loading cognitive models...",
            "Establishing neural link..."
        ]
        self.animation_event = Clock.schedule_interval(self._update_loading_animation, 0.05)

    def _update_loading_animation(self, dt):
        total_duration = 5.0
        self.animation_time += dt
        progress = int((self.animation_time / total_duration) * 20)
        progress = min(progress, 20)
        bar = "█" * progress + "░" * (20 - progress)
        self.loading_bar.text = f"[{bar}]"
        phase_duration = total_duration / (len(self.status_messages) + 1)
        
        if self.animation_time < phase_duration * (self.animation_phase + 1):
            if self.animation_phase < len(self.status_messages):
                current_message = self.status_messages[self.animation_phase]
                if self.typing_index < len(current_message):
                    self.typing_index += 1
                self.loading_status.text = f"> {current_message[:self.typing_index]}"
        else:
            self.animation_phase += 1
            self.typing_index = 0

        if self.animation_time >= total_duration:
            self.loading_status.text = "[color=00ffc8]> S Y N C   C O M P L E T E <[/color]"
            self.loading_status.markup = True
            self.animation_event.cancel()
            Clock.schedule_once(self.start_game, 1.0)

    def start_game(self, *args):
        if self.loading_layout.parent:
            self.remove_widget(self.loading_layout)
        if not self.camera:
            self.camera = Camera(play=True, resolution=(640, 480), opacity=0)
            self.add_widget(self.camera, index=10)
        if not self.detector:
            self.detector = GestureDetector()
            self.detector.start()
        self.add_widget(self.game_layout)
        self.game_layout.opacity = 1
        self.reset_game_state()

    def reset_game_state(self):
        self.score, self.lives, self.combo, self.zen_meter = 0, config.INITIAL_LIVES, 0, 0
        self.symbols, self.symbol_widgets = [], {}
        self.generate_new_rule()
        self.start_time = time.time()
        self.game_running, self.game_over = False, False
        self.instruction_panel.opacity = 1
        self.sync_zone_widget.opacity = 0
        for widget in list(self.symbol_widgets.values()):
            if widget.parent: self.play_area.remove_widget(widget)
        self.symbol_widgets.clear()
        if self.sound_start and (not self.sound_start.state == 'play'):
            self.sound_start.play()

    def restart_game(self, instance):
        if self.game_over_layout.parent:
            self.remove_widget(self.game_over_layout)
        self.game_layout.opacity = 1
        self.reset_game_state()

    def exit_game(self, instance):
        App.get_running_app().stop()

    def generate_new_rule(self):
        gestures = random.sample(config.GESTURES, len(config.SYMBOLS))
        self.current_rule = {symbol: gesture for symbol, gesture in zip(config.SYMBOLS, gestures)}
        rule_text = " | ".join([f"{s} -> {g.replace('_', ' ').replace('RAISED', 'UP')}" for s, g in self.current_rule.items()])
        self.rule_label.text = rule_text
        instr_text = "-- CONTROLS --\n\n" + "\n".join([f"{s}  ->  {g.replace('_', ' ').replace('RAISED', 'UP')}" for s, g in self.current_rule.items()]) + "\n\nStarting in..."
        self.instruction_panel.text = instr_text

    def update(self, dt):
        for star in self.stars:
            star['pos'][0] -= star['velocity'] * dt
            if star['pos'][0] < 0:
                star['pos'][0] = self.width
                star['pos'][1] = random.uniform(0, self.height)
            star['rect'].pos = star['pos']

        if self.camera and self.camera.texture:
            self.detector.update_frame(self.frame_from_texture(self.camera.texture))
        
        if self.game_over or not self.game_layout.opacity:
            return

        current_time = time.time()
        if self.lives <= 0:
            self.end_game()
            return
            
        time_since_start = current_time - self.start_time
        if time_since_start < config.STARTING_DELAY:
            countdown = int(config.STARTING_DELAY - time_since_start)
            self.instruction_panel.text = self.instruction_panel.text.rsplit(' ', 1)[0] + f" {countdown}"
            self.score_label.text = f"Score: {self.score}"
            self.lives_label.text = f"Lives: {self.lives}"
            return
        elif self.instruction_panel.opacity == 1:
            self.instruction_panel.opacity = 0
            self.game_running = True
            self.last_symbol_time = time.time()
            self.sync_zone_widget.opacity = 1

        if not self.game_running: return

        self.score_label.text = f"Score: {self.score}"
        self.lives_label.text = f"Lives: {self.lives}"
        self.combo_label.text = f"Combo: {self.combo}x"
        zen_bar = "█" * int(self.zen_meter / 10) + " " * (10 - int(self.zen_meter / 10))
        self.zen_label.text = f"Zen: [{zen_bar}]"

        if current_time - self.last_symbol_time > 2.0:
            self.spawn_symbol()
            self.last_symbol_time = current_time

        sync_zone_x_center = self.width * 0.5
        sync_zone_width = self.width * 0.25 
        gesture = self.detector.get_gesture()
        
        for symbol_data in list(self.symbols):
            widget = self.symbol_widgets.get(symbol_data['id'])
            if not widget: continue

            widget.x += config.STREAM_SPEED * dt * 10
            
            in_sync_zone = sync_zone_x_center - sync_zone_width / 2 < widget.center_x < sync_zone_x_center + sync_zone_width / 2
            
            if in_sync_zone and not symbol_data.get('in_zone', False):
                widget.color = (1, 1, 0, 1) # Yellow
                symbol_data['in_zone'] = True

            if in_sync_zone and not symbol_data['checked']:
                expected_gesture = self.current_rule[symbol_data['text']]
                
                if gesture == expected_gesture:
                    self.score += config.PERFECT_SCORE
                    self.combo += 1
                    self.zen_meter = min(config.ZEN_METER_MAX, self.zen_meter + config.ZEN_PER_HIT)
                    widget.color = (0, 1, 0, 1)
                    anim = Animation(font_size=50, duration=0.1) + Animation(font_size=40, duration=0.1)
                    anim.start(widget)
                    if self.sound_correct: self.sound_correct.play()
                    symbol_data['checked'] = True

                elif gesture != 'NONE':
                    self.lives -= 1
                    self.combo = 0
                    widget.color = (1, 0, 0, 1)
                    if self.sound_miss: self.sound_miss.play()
                    symbol_data['checked'] = True

            elif not in_sync_zone and symbol_data.get('in_zone', False):
                if not symbol_data['checked']:
                    self.lives -= 1
                    self.combo = 0
                    widget.color = (1,0,0,1)
                    if self.sound_miss: self.sound_miss.play()
                    symbol_data['checked'] = True
                elif widget.color != [0,1,0,1]:
                    widget.color = (1,1,1,1)


            if widget.x > self.width:
                if widget.parent: self.play_area.remove_widget(widget)
                self.symbols.remove(symbol_data)
                del self.symbol_widgets[symbol_data['id']]

    def spawn_symbol(self):
        symbol_text = random.choice(config.SYMBOLS)
        symbol_id = time.time() + random.random()
        symbol_data = {'id': symbol_id, 'text': symbol_text, 'checked': False, 'in_zone': False}
        self.symbols.append(symbol_data)
        widget = Label(text=symbol_text, font_name=config.FONT_NAME, font_size='40sp', size_hint=(None, None), size=('80sp', '80sp'), pos_hint={'center_y': 0.5}, x=-80)
        self.symbol_widgets[symbol_id] = widget
        self.play_area.add_widget(widget)

    def end_game(self):
        self.game_running, self.game_over = False, True
        self.game_layout.opacity = 0
        self.sync_zone_widget.opacity = 0
        for widget in list(self.symbol_widgets.values()):
            if widget.parent: self.play_area.remove_widget(widget)
        self.symbol_widgets.clear()
        self.symbols.clear()
        self.final_score_label.text = f"Final Score: {self.score}"
        self.add_widget(self.game_over_layout)
        if self.sound_start: self.sound_start.stop()
        if self.sound_gameover: self.sound_gameover.play()

    def frame_from_texture(self, texture):
        size = texture.size
        pixels = texture.pixels
        pil_image = np.frombuffer(pixels, dtype=np.uint8).reshape(size[1], size[0], 4)
        return cv2.cvtColor(pil_image, cv2.COLOR_RGBA2BGR)

class FocusFaceApp(App):
    def build(self):
        return GameWidget()

    def on_stop(self):
        if self.root and self.root.detector:
            self.root.detector.stop()

if __name__ == '__main__':
    FocusFaceApp().run()

