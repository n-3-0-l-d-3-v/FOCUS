# config.py
"""
Configuration file for game constants and settings.
"""

# -- UI and Styling --
FONT_NAME = 'assets/fonts/VT323-Regular.ttf'

# -- Game Logic Settings --
INITIAL_LIVES = 3
STREAM_SPEED = 15.0  # Characters per second
SYMBOLS = ['[X]', '[O]']
GESTURES = ['MOUTH_OPEN', 'EYEBROWS_UP'] 
STARTING_DELAY = 10 # Seconds before symbols start moving

# -- Scoring --
PERFECT_SCORE = 100
MISS_PENALTY = -25 # Not currently used, but here for future expansion
COMBO_BONUS = 10 # Not currently used, but here for future expansion

# -- Zen Meter --
ZEN_METER_MAX = 100
ZEN_PER_HIT = 10 # How much the meter fills per correct hit

# --- Gesture Detection Sensitivity ---
# You can fine-tune these values if gestures are not being detected reliably.
#
# To make MOUTH OPEN easier to detect, DECREASE this value (e.g., from 0.5 to 0.4).
MOUTH_AR_THRESHOLD = 0.45
# To make EYEBROWS UP easier to detect, DECREASE this value (e.g., from 0.08 to 0.075).
EYEBROW_RAISE_THRESHOLD = 0.075


# -- File Paths --
SOUND_CORRECT = 'assets/audio/sync_correct.wav'
SOUND_MISS = 'assets/audio/sync_miss.wav'
SOUND_START = 'assets/audio/background.mp3'
SOUND_GAMEOVER = 'assets/audio/flow_enter.mp3'

# -- Performance --
REFRESH_RATE = 1 / 30.0  # Target frame rate (30 FPS)

