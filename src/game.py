import math
import random

import pymunk
from pymunk import Body, Space, Vec2d

from game_status import GameStatus

BRICK_NB = 21 * 7

COLLISION_TYPES = {
    "ball": 1,
    "brick": 2,
    "bottom": 3,
    "player": 4,
}

class Game:
    _ball_body: Body
    _player_body: Body
    _score: int
    _space: Space

    def __init__(self):
        self._score = 0

        self._space = pymunk.Space()
        
        self._setup_walls()
        self._setup_player()
        self._setup_collisions()
        self.setup_level()

    def get_ball(self) -> Body | None:
        return self._ball_body

    def get_player(self) -> Body:
        return self._player_body

    def get_score(self) -> int:
        return self._score

    def get_game_status(self) -> GameStatus:
        if self._score >= BRICK_NB:
            return GameStatus.WON

        if self._ball_body.position.y <= 0:
            return GameStatus.LOST

        return GameStatus.ON_GOING

    def move_player_left(self):
        self._player_body.velocity = (-600, 0)

    def move_player_right(self):
        self._player_body.velocity = (600, 0)

    def stop_player(self):
        self._player_body.velocity = (0, 0)

    def update(self, dt: float):
        self._space.step(dt)

    def setup_level(self):
        self._remove_dynamic_bodies()
        self._score = 0

        self._spawn_ball(
            self._player_body.position + (0, 40),
            random.choice([(1, 10), (-1, 10)])
        )

        self._spawn_bricks()

    def _remove_dynamic_bodies(self):
        for shape in list(self._space.shapes):
            body = shape.body

            if (
                body.body_type == pymunk.Body.DYNAMIC
                and body != self._player_body
            ):
                self._space.remove(body, shape)


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
            if body.velocity.length > 0:
                body.velocity = body.velocity.normalized() * 400

        self._ball_body.velocity_func = constant_velocity

        self._space.add(self._ball_body, ball_shape)


    def _spawn_bricks(self):
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

    def _setup_walls(self):
        walls = [
            pymunk.Segment(
                self._space.static_body,
                (50, 50),
                (50, 550),
                2
            ),
            pymunk.Segment(
                self._space.static_body,
                (50, 550),
                (550, 550),
                2
            ),
            pymunk.Segment(
                self._space.static_body,
                (550, 550),
                (550, 50),
                2
            ),
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
        self._player_body = pymunk.Body(
            500,
            float("inf")
        )

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


        self._space.on_collision(
            COLLISION_TYPES["ball"],
            COLLISION_TYPES["bottom"],
        )

        def remove_brick(arbiter, space, data):
            brick_shape = arbiter.shapes[0]

            space.remove(
                brick_shape,
                brick_shape.body
            )

            self._score += 1

        self._space.on_collision(
            COLLISION_TYPES["brick"],
            COLLISION_TYPES["ball"],
            separate=remove_brick
        )

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
            COLLISION_TYPES["player"],
            COLLISION_TYPES["ball"],
            pre_solve=pre_solve
        )