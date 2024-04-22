import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("RoboVision Challenge")

clock = pygame.time.Clock()

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 5
        self.angle = 0  # Robot's current angle in degrees
        self.image = pygame.Surface((30, 30))  # Placeholder for robot's image
        self.rect = self.image.get_rect(center=(self.x, self.y))

    def move_towards_direction(self, direction):
        # Calculate movement vector based on robot's facing angle and direction
        angle_rad = np.radians(self.angle)
        if direction == 'w':
            angle_rad += np.radians(150)  # Move towards top-left
        elif direction == 'd':
            angle_rad -= np.radians(0)  # Move towards top-right
        elif direction == 'a':
            angle_rad += np.radians(190)  # Move towards left
        elif direction == 'z':
            angle_rad -= np.radians(30)  # Move towards bottom-left
        elif direction == 'x':
            angle_rad -= np.radians(0)  # Move towards bottom
        elif direction == 'e':
            angle_rad -= np.radians(150)  # Move towards top

        self.x += self.speed * np.cos(angle_rad)
        self.y -= self.speed * np.sin(angle_rad)
        self.rect.center = (self.x, self.y)

    def turn(self, direction):
        # Update robot's angle (0 = east, 90 = north, etc.)
        if direction == 'left':
            self.angle += 90
        elif direction == 'right':
            self.angle -= 90
        self.angle %= 360

    def draw(self, surface):
        # Rotate the robot's image based on the current angle
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        surface.blit(rotated_image, self.rect)

class Game:
    def __init__(self):
        self.robot = Robot(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.robot.move_towards_direction('w')
                elif event.key == pygame.K_d:
                    self.robot.move_towards_direction('d')
                elif event.key == pygame.K_a:
                    self.robot.move_towards_direction('a')
                elif event.key == pygame.K_z:
                    self.robot.move_towards_direction('z')
                elif event.key == pygame.K_x:
                    self.robot.move_towards_direction('x')
                elif event.key == pygame.K_e:
                    self.robot.move_towards_direction('e')

    def update(self):
        # Update game logic
        pass

    def render(self):
        screen.fill(WHITE)
        self.robot.draw(screen)
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            clock.tick(FPS)

if __name__ == "__main__":
    game = Game()
    game.run()

pygame.quit()
