import pygame
import time

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()


def get_controller_inputs(joystick):
    for event in pygame.event.get():
        pass

    v = -joystick.get_axis(4)
    w = -joystick.get_axis(3)
    b = joystick.get_button(0) 
    return v, w, b


for i in range(400):
    v, w, b = get_controller_inputs(joystick)
    time.sleep(0.039)
    print(v, w, b)