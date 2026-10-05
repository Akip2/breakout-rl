import sys, time
from game import Game
from game_renderer import GameRenderer

class GameLoop:
    _headless: bool
    _fps: float
    _running: bool

    def __init__(self, fps: float = 60, headless = False):
        self._headless = headless
        self._fps = fps
        self._running = False

    def run(self): 
        game = Game()  
        dt = 1.0 / self._fps

        if(not self._headless):
            game_renderer = GameRenderer(game=game, fps=self._fps)

        self._running = True
        while self._running:
            if(not self._headless): game_renderer.handle_events()

            game.update(dt)

            if(not self._headless):
                game_renderer.render()
                self._running = game_renderer.is_running()
                game_renderer.tick()
            else:
                time.sleep(sys.float_info.min) # allow ctr-c termination in headless mode