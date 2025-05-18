import math
from config import BALL_RADIUS, BALL_MASS, FRICTION_COEFF, COLLISION_DAMPING, POCKET_RADIUS
from physics.spin import apply_spin_force
from physics.table_surface import get_curvature_force

class Ball:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.mass = BALL_MASS
        self.radius = BALL_RADIUS

    def apply_velocity(self, vx, vy):
        self.vx = vx
        self.vy = vy
    
    def update_position(self, dt, table_width, table_height):
        # Apply friction
        friction = FRICTION_COEFF
        speed = math.hypot(self.vx, self.vy)
        # Apply table warp
        fx, fy = get_curvature_force(self.x, self.y)
        self.vx += fx * dt
        self.vy += fy * dt

        if speed > 0:
            decel = friction * 9.81
            speed = max(0, speed - decel * dt)
            angle = math.atan2(self.vy, self.vx)
            self.vx = speed * math.cos(angle)
            self.vy = speed * math.sin(angle)
        
        # Move ball
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce of table edges
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx *= -COLLISION_DAMPING
        elif self.x + self.radius > table_width:
            self.x = table_width - self.radius
            self.vx *= -COLLISION_DAMPING
        
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy *= -COLLISION_DAMPING
        elif self.y + self.radius > table_height:
            self.y = table_height - self.radius
            self.vy *= -COLLISION_DAMPING

    def is_stopped(self):
        return math.hypot(self.vx, self.vy) < 0.01

class PoolSimulation:
    def __init__(self, table_width, table_height):
        self.width = table_width
        self.height = table_height
        self.cue_ball = Ball(table_width / 2, table_height / 2, BALL_RADIUS)
        self.object_ball = Ball(self.width / 3, self.height / 2, BALL_RADIUS)
        self.replay_buffer = []
    
    def step(self, dt):
        self.replay_buffer.append(self.get_ball_states())
        self.cue_ball.update_position(dt, self.width, self.height)
        self.object_ball.update_position(dt, self.width, self.height)
        self.handle_collision(self.cue_ball, self.object_ball)
    
    def get_replay(self):
        return self.replay_buffer 
    
    def reset_balls(self):
        self.object_ball.x = self.width / 3
        self.object_ball.y = self.height / 2
        self.cue_ball.vx = 0
        self.cue_ball.vy = 0
    
    def apply_shot(self, angle_deg, speed, spin='none'):
        angle_rad = math.radians(angle_deg)
        self.cue_ball.apply_velocity(vx, vy)
        apply_spin_force(self.cue_ball, spin)

        # Apply squirt connection
        if spin == 'left':
            angle_rad -= math.radians(1)
        elif spin == 'right':
            angle_rad += math.radians(1)
        
        # Speed adjustment for spin
        if spin == 'top':
            speed *= 1.05
        elif spin == 'back':
            speed *= 0.95

        vx = speed * math.cos(angle_rad)
        vy = speed * math.sin(angle_rad)
        self.cue_ball.apply_velocity(vx, vy)
    
    def step(self, dt):
        self.cue_ball.update_position(dt, self.width, self.height)
        self.object_ball.update_position(dt, self.width, self.height)
        self.handle_collision(self.cue_ball, self.object_ball)
        self.cue_ball_sunk = False
        self.object_ball_sunk = False

        if not self.cue_ball_sunk and self.is_ball_sunk(self.cue_ball):
            self.cue_ball_sunk = True
            self.cue_ball.vx = self.cue_ball.vy = 0

        if not self.object_ball_sunk and self.is_ball_sunk(self.object_ball):
            self.object_ball_sunk = True
            self.object_ball.vx = self.object_ball.vy = 0 

    def is_ball_sunk(self, ball):
        pockets = [
            (0, 0),
            (self.width / 2, 0),
            (self.width, 0),
            (0, self.height),
            (self.width / 2, self.height),
            (self.width, self.height)
        ]
        for px, py in pockets:
            if math.hypot(ball.x - px, ball.y - py) < POCKET_RADIUS:
                return True
        return False
     
    def handle_collision(self, b1, b2):
        dx = b2.x - b1.x
        dy = b2.y - b1.y
        dist = math.hypot(dx, dy)

        if dist == 0 or dist > b1.radius + b2.radius:
            return # no collision
        
        # Normalize direction
        nx = dx / dist
        ny = dy / dist

        # Relative velocity
        dvx = b1.vx - b2.vx
        dvy = b1.vy - b2.vy
        rel_vel = dvx * nx + dvy * ny

        if rel_vel > 0:
            return # balls are separating
        
        # Momentum exchange (equal mass)
        impulse = -2 * rel_vel / (b1.mass + b2.mass)
        b1.vx += impulse * b2.mass * nx
        b1.vy += impulse * b2.mass * ny
        b2.vx -= impulse * b1.mass * nx
        b2.vy -= impulse * b1.mass * ny

        # Prevent overlap
        overlap = b1.radius + b2.radius - dist
        b1.x -= nx * overlap / 2
        b1.y -= ny * overlap / 2
        b2.x += nx * overlap / 2
        b2.y += ny * overlap / 2
    
    def all_stopped(self):
        return self.cue_ball.is_stopped() and self.object_ball.is_stopped()

    def get_ball_states(self):
        states = []
        if not self.cue_ball_sunk:
            states.append((self.cue_ball.x, self.cue_ball.y, self.cue_ball.radius))
        if not self.object_ball_sunk:
            states.append((self.object_ball.x, self.object_ball.y, self.object_ball.radius))
        return states
    
        return [
            (self.cue_ball.x, self.cue_ball.y, self.cue_ball.radius),
            (self.object_ball.x, self.object_ball.y, self.object_ball.radius)
        ]

