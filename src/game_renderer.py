import pygame
from pygame.font import Font
from pygame.time import Clock
import pymunk
import pymunk.pygame_util

from game import Game
from game_status import GameStatus

WIDTH = 600
HEIGHT = 600

class GameRenderer:
    _game: Game
    _screen: pygame.Surface
    _clock: Clock
    _running: bool
    _font: Font
    _clock: Clock
    _fps: float
    
    def __init__(self, game: Game, fps: float):
        self._game = game
        self._fps = fps

        pygame.init()

        self._screen = pygame.display.set_mode(
            (WIDTH, HEIGHT)
        )

        pygame.display.set_caption("Breakout")

        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Arial", 16)

        pymunk.pygame_util.positive_y_is_up = True

        self._draw_options = pymunk.pygame_util.DrawOptions(
            self._screen
        )

        self._running = True

        self._setup_colors()

    def is_running(self):
        return self._running
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._stop()

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_LEFT:
                    self._game.move_player_left()

                elif event.key == pygame.K_RIGHT:
                    self._game.move_player_right()

                elif event.key in (
                    pygame.K_ESCAPE,
                    pygame.K_q
                ):
                    self._running = False

            elif event.type == pygame.KEYUP:

                if event.key in (
                    pygame.K_LEFT,
                    pygame.K_RIGHT
                ):
                    self._game.stop_player()

    def _setup_colors(self):
        self.ball_color = pygame.Color("green")
        self.player_color = pygame.Color("red")
        self.brick_color = pygame.Color("blue")
        self.wall_color = pygame.Color("lightgray")

        self.background_color = pygame.Color("black")
        self.text_color = pygame.Color("white")
        self.help_color = pygame.Color("darkgrey")

    def render(self):
        self._screen.fill(self.background_color)

        self._apply_colors()

        self._game._space.debug_draw(
            self._draw_options
        )

        self._draw_score()
        self._draw_status()

        pygame.display.flip()

    def tick(self):
        self._clock.tick(self._fps)

    def _apply_colors(self):
        for shape in self._game._space.shapes:
            if shape.collision_type == 1:
                shape.color = self.ball_color

            elif shape.collision_type == 2:
                shape.color = self.brick_color

            elif shape.collision_type == 3:
                shape.color = pygame.Color("red")

            elif shape.collision_type == 4:
                shape.color = self.player_color

            else:
                shape.color = self.wall_color

    def _draw_score(self):
        text = self._font.render(
            f"Score: {self._game.get_score()}",
            True,
            self.text_color
        )

        self._screen.blit(text, (10, 10))

    def _draw_status(self):
        status = self._game.get_game_status()

        if status == GameStatus.WON:
            message = "YOU WON"

        elif status == GameStatus.LOST:
            message = "GAME OVER"

        else:
            return

        text = self._font.render(
            message,
            True,
            self.text_color
        )

        rect = text.get_rect(
            center=(WIDTH // 2, HEIGHT // 2)
        )

        self._screen.blit(text, rect)

    def _stop(self):
        self._running = False
        pygame.quit()