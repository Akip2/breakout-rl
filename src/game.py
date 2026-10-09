import math
import random

import pymunk
from pymunk import Body, Space, Vec2d

from game_status import GameStatus

BRICK_NB = 0  # 21 * 7

BALL_SPEED = 400
MIN_VERTICAL = 0.25
FLOOR_Y = 50

COLLISION_TYPES = {
    "ball": 1,
    "brick": 2,
    "bottom": 3,
    "player": 4,
}

class Game:
    _ball_body: Body
    _player_body: Body
    _hits: int
    _just_hit: bool
    _space: Space
    _bricks: list[Body]
    _nb_steps: int

    def __init__(self):
        self._hits = 0
        self._just_hit = False
        self._nb_steps = 0
        self._bricks = []

        self._space = pymunk.Space()
        self._setup()

    def _setup(self):
        self._setup_walls()
        self._setup_player()
        self._setup_collisions()
        self._setup_level()

    def reset(self):
        self._player_body.position = 300, 100
        self._player_body.velocity = (0, 0)

        self._hits = 0
        self._just_hit = False
        self._nb_steps = 0

        self._remove_dynamic_bodies()
        self._setup_level()

    def get_ball(self) -> Body:
        return self._ball_body

    def get_player(self) -> Body:
        return self._player_body

    def get_score(self) -> int:
        """Nombre de rebonds sur la raquette pendant la partie."""
        return self._hits

    def get_game_status(self) -> GameStatus:
        if self._ball_body.position.y < FLOOR_Y:
            return GameStatus.LOST

        return GameStatus.ON_GOING

    def get_reward(self) -> float:
        if self.get_game_status() == GameStatus.LOST:
            return -1.0

        reward = 0.0
        if self._just_hit:
            self._just_hit = False
            reward += 1.0

        ball = self._ball_body
        if ball.velocity.y < 0:
            dx = abs(ball.position.x - self._player_body.position.x)
            reward += 0.005 * max(0.0, 1.0 - dx / 250)

        return reward

    def move_player_left(self):
        if self._player_body.position.x <= 100:
            self.stop_player()
            return
        self._player_body.velocity = (-600, 0)

    def move_player_right(self):
        if self._player_body.position.x >= 500:
            self.stop_player()
            return
        self._player_body.velocity = (600, 0)

    def stop_player(self):
        self._player_body.velocity = (0, 0)

    def update(self, dt: float):
        self._nb_steps += 1
        self._space.step(dt)

    def _setup_level(self):
        self._spawn_ball(
            self._player_body.position + (0, 40),
            (random.uniform(-7, 7), 10),
        )

        self._spawn_bricks()

    def _remove_dynamic_bodies(self):
        if self._ball_body in self._space.bodies:
            self._space.remove(
                self._ball_body,
                *self._ball_body.shapes
            )

        for brick_body in self._bricks:
            if brick_body in self._space.bodies:
                self._space.remove(
                    brick_body,
                    *brick_body.shapes
                )

        self._bricks.clear()

    def _spawn_ball(self, position, direction):
        self._ball_body = pymunk.Body(1, float("inf"))
        self._ball_body.position = position

        ball_shape = pymunk.Circle(self._ball_body, 5)
        ball_shape.elasticity = 1.0
        ball_shape.collision_type = COLLISION_TYPES["ball"]

        self._ball_body.apply_impulse_at_local_point(
            Vec2d(*direction)
        )

        def constant_velocity(body, gravity, damping, dt):
            v = body.velocity
            if v.length == 0:
                return

            v = v.normalized()

            if abs(v.y) < MIN_VERTICAL:
                sign_y = 1 if v.y >= 0 else -1
                sign_x = 1 if v.x >= 0 else -1
                v = Vec2d(
                    sign_x * math.sqrt(1 - MIN_VERTICAL ** 2),
                    sign_y * MIN_VERTICAL,
                )

            body.velocity = v * BALL_SPEED

        self._ball_body.velocity_func = constant_velocity

        self._space.add(self._ball_body, ball_shape)

    def _spawn_bricks(self):
        self._bricks.clear()

        for i in range(BRICK_NB):
            y = math.floor(i / 21)
            x = i % 21

            brick_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            brick_body.position = 80 + x * 22, 400 + y * 22

            brick_shape = pymunk.Poly.create_box(
                brick_body,
                (20, 20)
            )

            brick_shape.elasticity = 1.0
            brick_shape.group = 1
            brick_shape.collision_type = COLLISION_TYPES["brick"]

            self._space.add(brick_body, brick_shape)
            self._bricks.append(brick_body)

    def _setup_walls(self):
        walls = [
            pymunk.Segment(self._space.static_body, (50, 50), (50, 550), 2),
            pymunk.Segment(self._space.static_body, (50, 550), (550, 550), 2),
            pymunk.Segment(self._space.static_body, (550, 550), (550, 50), 2),
        ]

        for wall in walls:
            wall.elasticity = 1.0

        bottom = pymunk.Segment(
            self._space.static_body,
            (50, 50),
            (550, 50),
            2
        )

        bottom.sensor = True
        bottom.collision_type = COLLISION_TYPES["bottom"]

        self._space.add(*walls)
        self._space.add(bottom)

    def _setup_player(self):
        self._player_body = pymunk.Body(500, float("inf"))
        self._player_body.position = 300, 100

        player_shape = pymunk.Segment(
            self._player_body,
            (-50, 0),
            (50, 0),
            8
        )

        player_shape.elasticity = 1.0
        player_shape.collision_type = COLLISION_TYPES["player"]

        move_joint = pymunk.GrooveJoint(
            self._space.static_body,
            self._player_body,
            (100, 100),
            (500, 100),
            (0, 0)
        )

        self._space.add(
            self._player_body,
            player_shape,
            move_joint
        )

        self._player_shape = player_shape

    def _setup_collisions(self):
        def remove_brick(arbiter, space, data):
            brick_shape = arbiter.shapes[0]

            space.remove(
                brick_shape,
                brick_shape.body
            )

        def on_player_hit(arbiter, space, data):
            self._hits += 1
            self._just_hit = True

        def pre_solve(arbiter, space, data):
            contact_set = arbiter.contact_point_set

            if len(contact_set.points) > 0:
                player_shape = arbiter.shapes[0]

                width = (
                    player_shape.b -
                    player_shape.a
                ).x

                delta = (
                    player_shape.body.position -
                    contact_set.points[0].point_a
                ).x

                normal = Vec2d(0, 1).rotated(
                    delta / width / 2
                )

                contact_set.normal = normal
                contact_set.points[0].distance = 0

            arbiter.contact_point_set = contact_set

        self._space.on_collision(
            COLLISION_TYPES["brick"],
            COLLISION_TYPES["ball"],
            separate=remove_brick
        )

        self._space.on_collision(
            COLLISION_TYPES["player"],
            COLLISION_TYPES["ball"],
            begin=on_player_hit,
            pre_solve=pre_solve
        )
