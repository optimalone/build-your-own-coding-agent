"""
snake.py — Classic Snake game built with Pygame.
Single-file implementation following PLAN.md.
"""

import sys
import random
import pygame

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

WINDOW_W, WINDOW_H = 600, 600
COLS, ROWS         = 20, 20
CELL               = WINDOW_W // COLS       # 30 px
FPS                = 10

# Colours
BG_COLOUR          = (26,  26,  46)         # #1a1a2e — dark charcoal
GRID_COLOUR        = (36,  36,  60)         # subtle grid lines
HEAD_COLOUR        = (0,   230,  50)        # bright lime
BODY_COLOUR        = (0,   160,  30)        # darker green
FOOD_COLOUR        = (220,  30,  30)        # bright red
FOOD_SHINE         = (255, 120, 120)        # highlight dot on food
TEXT_WHITE         = (240, 240, 240)
TEXT_YELLOW        = (255, 220,  50)
TEXT_RED           = (220,  30,  30)
OVERLAY_COLOUR     = (10,  10,  25, 180)    # semi-transparent panel (RGBA)

# HUD layout
HUD_H              = 0                      # score is rendered inside the grid area
TOP_PADDING        = 4                      # px gap between HUD text and grid top

# ---------------------------------------------------------------------------
# Snake
# ---------------------------------------------------------------------------

class Snake:
    """Manages the snake body, direction, movement, and rendering."""

    START_COL = COLS // 2
    START_ROW = ROWS // 2

    def __init__(self):
        self.reset()

    def reset(self):
        """Restore snake to initial 3-segment horizontal state."""
        self.body           = [
            (self.START_COL,     self.START_ROW),
            (self.START_COL - 1, self.START_ROW),
            (self.START_COL - 2, self.START_ROW),
        ]
        self.direction      = (1, 0)   # moving right
        self.next_direction = (1, 0)
        self.grow_pending   = False

    def change_direction(self, dx: int, dy: int):
        """Queue a direction change; block 180° reversals."""
        cur_dx, cur_dy = self.direction
        if (dx, dy) != (-cur_dx, -cur_dy):
            self.next_direction = (dx, dy)

    def move(self):
        """Advance the snake by one cell, grow if food was just eaten."""
        self.direction = self.next_direction
        head_col, head_row = self.body[0]
        dx, dy = self.direction
        new_head = (head_col + dx, head_row + dy)
        self.body.insert(0, new_head)
        if self.grow_pending:
            self.grow_pending = False   # keep the tail segment → snake grows
        else:
            self.body.pop()            # remove tail → length unchanged

    def draw(self, surface: pygame.Surface):
        """Render each body segment; head is a brighter colour."""
        for i, (col, row) in enumerate(self.body):
            colour = HEAD_COLOUR if i == 0 else BODY_COLOUR
            rect   = pygame.Rect(col * CELL, row * CELL, CELL, CELL)
            pygame.draw.rect(surface, colour, rect, border_radius=6)
            # Inner highlight ring on head
            if i == 0:
                inner = rect.inflate(-8, -8)
                pygame.draw.rect(surface, (120, 255, 140), inner, width=2, border_radius=4)

# ---------------------------------------------------------------------------
# Food
# ---------------------------------------------------------------------------

class Food:
    """Manages the food pellet position and rendering."""

    def __init__(self, snake_body: list):
        self.pos = (0, 0)
        self.respawn(snake_body)

    def respawn(self, snake_body: list):
        """Pick a random grid cell not occupied by the snake."""
        occupied = set(snake_body)
        all_cells = [
            (c, r) for c in range(COLS) for r in range(ROWS)
            if (c, r) not in occupied
        ]
        self.pos = random.choice(all_cells)

    def draw(self, surface: pygame.Surface):
        """Render the food as a red circle with a small shine dot."""
        col, row = self.pos
        cx = col * CELL + CELL // 2
        cy = row * CELL + CELL // 2
        radius = CELL // 2 - 3
        pygame.draw.circle(surface, FOOD_COLOUR, (cx, cy), radius)
        # Shine highlight
        pygame.draw.circle(surface, FOOD_SHINE, (cx - radius // 3, cy - radius // 3), radius // 4)

# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class Game:
    """Top-level controller: state machine, event handling, update, draw."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("🐍 Snake")
        self.screen     = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock      = pygame.time.Clock()
        self.font_hud   = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_big   = pygame.font.SysFont("consolas", 52, bold=True)
        self.font_mid   = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 20)
        self.high_score = 0
        self.snake      = Snake()
        self.food       = Food(self.snake.body)
        self.score      = 0
        self.state      = "playing"

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self):
        """Reset snake, food, and score; transition back to playing."""
        self.snake.reset()
        self.food.respawn(self.snake.body)
        self.score = 0
        self.state = "playing"

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()

            elif event.type == pygame.KEYDOWN:
                key = event.key

                # Quit anytime
                if key == pygame.K_ESCAPE:
                    self._quit()

                # Movement — playing only
                if self.state == "playing":
                    if key in (pygame.K_UP,    pygame.K_w): self.snake.change_direction( 0, -1)
                    if key in (pygame.K_DOWN,  pygame.K_s): self.snake.change_direction( 0,  1)
                    if key in (pygame.K_LEFT,  pygame.K_a): self.snake.change_direction(-1,  0)
                    if key in (pygame.K_RIGHT, pygame.K_d): self.snake.change_direction( 1,  0)

                # Restart — game over screen only
                if self.state == "game_over":
                    if key in (pygame.K_r, pygame.K_RETURN):
                        self.reset()

    @staticmethod
    def _quit():
        pygame.quit()
        sys.exit()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self):
        """Advance game logic one tick (called only while playing)."""
        if self.state != "playing":
            return

        self.snake.move()
        head = self.snake.body[0]
        col, row = head

        # Wall collision
        if not (0 <= col < COLS and 0 <= row < ROWS):
            self._end_game()
            return

        # Self collision (head vs rest of body)
        if head in self.snake.body[1:]:
            self._end_game()
            return

        # Food collision
        if head == self.food.pos:
            self.score += 1
            if self.score > self.high_score:
                self.high_score = self.score
            self.snake.grow_pending = True
            self.food.respawn(self.snake.body)

    def _end_game(self):
        self.state = "game_over"

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self):
        # 1. Background
        self.screen.fill(BG_COLOUR)

        # 2. Grid lines
        self._draw_grid()

        # 3. Food & snake
        self.food.draw(self.screen)
        self.snake.draw(self.screen)

        # 4. HUD
        self._draw_hud()

        # 5. Game over overlay (if applicable)
        if self.state == "game_over":
            self._draw_game_over()

        pygame.display.flip()

    def _draw_grid(self):
        for c in range(COLS + 1):
            x = c * CELL
            pygame.draw.line(self.screen, GRID_COLOUR, (x, 0), (x, WINDOW_H))
        for r in range(ROWS + 1):
            y = r * CELL
            pygame.draw.line(self.screen, GRID_COLOUR, (0, y), (WINDOW_W, y))

    def _draw_hud(self):
        # Score — top-left
        score_surf = self.font_hud.render(f"Score: {self.score}", True, TEXT_YELLOW)
        self.screen.blit(score_surf, (8, 6))

        # High score — top-right
        hs_surf = self.font_hud.render(f"Best: {self.high_score}", True, TEXT_WHITE)
        self.screen.blit(hs_surf, (WINDOW_W - hs_surf.get_width() - 8, 6))

    def _draw_game_over(self):
        # Semi-transparent overlay panel
        panel_w, panel_h = 420, 240
        panel_x = (WINDOW_W - panel_w) // 2
        panel_y = (WINDOW_H - panel_h) // 2

        overlay = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        overlay.fill(OVERLAY_COLOUR)
        self.screen.blit(overlay, (panel_x, panel_y))

        # Thin border around panel
        pygame.draw.rect(self.screen, TEXT_RED, (panel_x, panel_y, panel_w, panel_h), width=2, border_radius=8)

        cx = WINDOW_W // 2

        # "GAME OVER"
        go_surf = self.font_big.render("GAME OVER", True, TEXT_RED)
        self.screen.blit(go_surf, go_surf.get_rect(centerx=cx, top=panel_y + 18))

        # Final score
        fs_surf = self.font_mid.render(f"Score: {self.score}", True, TEXT_YELLOW)
        self.screen.blit(fs_surf, fs_surf.get_rect(centerx=cx, top=panel_y + 88))

        # Restart prompt
        r_surf = self.font_small.render("Press  R  or  Enter  to Restart", True, TEXT_WHITE)
        self.screen.blit(r_surf, r_surf.get_rect(centerx=cx, top=panel_y + 148))

        # Quit prompt
        q_surf = self.font_small.render("Press  ESC  to Quit", True, TEXT_WHITE)
        self.screen.blit(q_surf, q_surf.get_rect(centerx=cx, top=panel_y + 180))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Game().run()
