import pygame
import sys
import math

# Define Ball class and BALL_RADIUS if not imported from elsewhere
BALL_RADIUS = 0.028575  # Standard pool ball radius in meters (example value)

class Ball:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

# color palette
GREEN = (20, 120, 20)
WHITE = (255, 255, 255)

class PoolVisualizer:
    def __init__(self, table_width, table_height, pocket_radius):
        pygame.init()
        self.scale = 400 # pixels per meter
        self.margin = 50
        width_px = int(table_width * self.scale + 2 * self.margin)
        height_px = int(table_height * self.scale + 2 * self.margin)
        self.start_pos = None
        self.end_pos = None
        self.shot_ready = False

        self.screen = pygame.display.set_mode((width_px, height_px))
        pygame.display.set_caption("Precision Pool Simulator")
        self.clock = pygame.time.Clock()
        self.table_width = table_width
        self.table_height = table_height
        self.pocket_radius = pocket_radius
    
    def wait_for_shot_input(self, cue_ball_pos):
        self.shot_ready = False
        self.start_pos = None
        self.end_pos = None

        while not self.shot_ready:
            self.draw([cue_ball_pos])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.start_pos = pygame.mouse.get_pos()
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.end_pos = pygame.mouse.get_pos()
                    self.shot_ready = True
        
        # Convert drag to angle and speed
        dx = self.end_pos[0] - self.start_pos[0]
        dy = self.end_pos[1] - self.start_pos[1]

        angle = math.degrees(math.atan2(dy, dx))
        speed = math.hypot(dx, dy) / 80  # scaling factor
        return {
            "angle_deg": angle,
            "speed": speed,
            "spin": "none"  # For now, no spin control here
        }
    
    def play_replay(self, replay_frames):
        for frame in replay_frames:
            self.draw(frame)
            pygame.time.delay(15)
    
    def draw(self, ball_states):
        self.screen.fill((0, 0, 0)) # Background
        pygame.draw.rect(
            self.screen, GREEN,
            (self.margin, self.margin, self.table_width * self.scale, self.table_height * self.scale)
        )

        for x, y, r in ball_states:
            px = self.margin + int(x * self.scale)
            py = self.margin + int(y * self.scale)
            pr = int(r * self.scale)
            pygame.draw.circle(self.screen, WHITE, (px, py), pr)
        
        pygame.display.flip()
        self.clock.tick(60)

        # Quit Logic
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()