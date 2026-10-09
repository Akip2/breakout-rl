import sys
import time

from agent import Agent
from game import Game
from game_renderer import GameRenderer
from game_status import GameStatus

FRAME_SKIP = 4 
class GameLoop:
    _headless: bool
    _fast: bool
    _fps: float
    _running: bool

    def __init__(self, fps: float = 60, headless=False, fast: bool = False):
        self._headless = headless
        self._fps = fps
        self._running = False
        self._fast = fast

    def _apply_action(self, game: Game, action: int):
        if action == 0:
            game.move_player_left()
        elif action == 1:
            game.stop_player()
        else:
            game.move_player_right()

    def run(self):
        game = Game()
        agent = Agent(game=game)
        record = 0

        dt = 1.0 / self._fps

        if not self._headless:
            game_renderer = GameRenderer(game=game, fps=self._fps)

        state = agent.get_state()
        ep_decisions = 0
        ep_reward = 0.0
        losses = []

        self._running = True
        while self._running:
            if not self._headless:
                game_renderer.handle_events()

            action = agent.get_action(state)
            self._apply_action(game, action)

            reward = 0.0
            status = GameStatus.ON_GOING
            for i in range(FRAME_SKIP):
                game.update(dt)
                reward += game.get_reward()
                status = game.get_game_status()

                if not self._headless:
                    game_renderer.render()
                    self._running = game_renderer.is_running()

                if(not self._fast):
                    if(self._headless):
                        time.sleep(1 / self._fps)
                    else:
                        game_renderer.tick()
                else:
                    time.sleep(sys.float_info.min) # allow ctrl-c quit

                if status != GameStatus.ON_GOING:
                    break

            next_state = agent.get_state()
            done = status == GameStatus.LOST

            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                losses.append(loss)

            state = next_state
            ep_decisions += 1
            ep_reward += reward

            if done:
                hits = game.get_score()
                avg_loss = sum(losses) / len(losses) if losses else float("nan")
                print(
                    f"game {agent._nb_games:5d} | hits {hits:3d} | "
                    f"reward {ep_reward:7.2f} | eps {agent.epsilon:.2f} | "
                    f"loss {avg_loss:.4f} | record {record}"
                )

                if hits > record:
                    record = hits
                    agent.save_model()

                game.reset()
                agent.new_game()
                state = agent.get_state()
                ep_decisions = 0
                ep_reward = 0.0
                losses = []