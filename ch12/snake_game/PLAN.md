# 🐍 Snake Game — Plan

## Overview
A classic Snake game built with Python and Pygame, all contained in a single file: `snake.py`.

---

## File Structure
```
snake.py      # All game code lives here
PLAN.md       # This plan
```

---

## Architecture

### Constants & Configuration
- Window size: 600×600 px
- Grid size: 20×20 cells (each cell = 30px)
- FPS: 10 (classic snake feel)
- Color palette: dark background, bright snake, red food, UI text

### Classes

#### `Snake`
- `body`: list of (col, row) tuples — head is index 0
- `direction`: current movement vector (dx, dy)
- `next_direction`: buffered input to avoid mid-frame reversals
- `grow_pending`: flag set when food is eaten
- `reset()`: restore to starting state
- `change_direction(dx, dy)`: queue a direction change, block 180° reversal
- `move()`: advance head, optionally grow tail
- `draw(surface)`: render each segment; distinct color for head

#### `Food`
- `pos`: (col, row) tuple
- `respawn(snake_body)`: pick a random grid cell not occupied by the snake
- `draw(surface)`: render the food pellet

#### `Game`
- `state`: `"playing"` | `"game_over"`
- `score`: integer, incremented on each food eaten
- `high_score`: tracked across restarts (session only)
- `snake`: Snake instance
- `food`: Food instance
- `reset()`: reset snake, food, score; set state to playing
- `handle_events()`: process quit, arrow keys / WASD, restart key (R or Enter)
- `update()`: move snake, check wall/self collision, check food collision
- `draw()`: render background grid, food, snake, HUD (score), or Game Over overlay
- `run()`: main loop — event → update → draw → tick

### Game Loop Flow
```
run()
 └─ while True
      ├─ handle_events()   # input & state transitions
      ├─ update()          # only when state == "playing"
      └─ draw()            # always
           ├─ draw grid
           ├─ food.draw()
           ├─ snake.draw()
           ├─ draw HUD (score / high-score)
           └─ if game_over → draw overlay + restart prompt
```

---

## Screens

### Playing HUD
- Top bar: `Score: X`  |  `High Score: X`

### Game Over Overlay
- Semi-transparent dark panel centred on screen
- "GAME OVER" in large red text
- Final score
- "Press R or Enter to Restart"
- "Press ESC to Quit"

---

## Controls
| Key | Action |
|-----|--------|
| Arrow Keys / WASD | Move snake |
| R / Enter | Restart (Game Over screen) |
| ESC | Quit |

---

## Collision Detection
- **Wall collision**: head position outside grid bounds → game over
- **Self collision**: head position appears in body list (index 1+) → game over
- **Food collision**: head position == food position → grow + respawn food + increment score

---

## Visual Details
- Background: dark charcoal (`#1a1a2e`)
- Grid lines: subtle dark lines for depth
- Snake body: green gradient — head is bright lime, body is darker green
- Food: bright red circle with a small shine dot
- Text: white / yellow for HUD; red for Game Over title

---

## Implementation Steps
1. Imports & constants
2. `Snake` class
3. `Food` class
4. `Game` class — `reset`, `handle_events`, `update`, `draw`, `run`
5. `if __name__ == "__main__"` entry point
