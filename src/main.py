from game_loop import GameLoop

if __name__ == "__main__":
    game_loop = GameLoop(fps=60, headless=False, fast=False)
    game_loop.run()