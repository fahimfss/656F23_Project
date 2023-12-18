import argparse
import shutil
import time
import os
import numpy as np
import pygame


from reacher_env import CarterReacherEnv
import utils
from utils import WrappedEnv
from replay_buffer import RadReplayBuffer
import cv2


def get_controller_inputs(joystick):
    for event in pygame.event.get():
        pass

    v = -joystick.get_axis(4)
    w = -joystick.get_axis(3)
    b = joystick.get_button(0) 

    return v, w, b

def show_state(img):
    cv2.imshow('im', img)
    cv2.waitKey(1)


def main():
    pygame.init()
    pygame.joystick.init()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    dir = 'reacher_st/hf/'
    env = CarterReacherEnv(scene_path=dir+'arena2.usd',
                           seed=56,
                           image_width=160,
                           image_height=90,
                           headless=False,
                           img_type='chw')
    env = WrappedEnv(env,
                     episode_max_steps=500, 
                     start_step=0, 
                     start_episode=0)

    image_shape = env.image_space.shape
    proprioception_shape = env.proprioception_space.shape
    action_shape = env.action_space.shape

    buffer = RadReplayBuffer(image_shape, proprioception_shape, action_shape, 
                             20000, 500)

    
    episode, episode_reward, episode_step, done = 0, 0, 0, True

    returns = []
    epi_lens = []

    task_start_time = time.time()

    hf_img, image, propri = env.reset()
    show_state(hf_img)

    while True:
        _, _, b = get_controller_inputs(joystick)
        if b == 1:
            break
        else:
            time.sleep(0.01)

    while episode < 50:
        show_state(hf_img)
        v, w, _ = get_controller_inputs(joystick)

        action = np.array([v, w])

        (hf_img, next_image, next_propri), reward, done, info = env.step(action)

        episode_reward += reward
        episode_step += 1

        if not done or 'TimeLimit.truncated' in info:
            mask = 0.0
        else:
            mask = 1.0

        buffer.add(image, propri, action, reward, next_image, next_propri, mask)
        
        if done or (episode_step == 500): # set time out here
            elapsed_time = "{:.3f}".format(time.time() - task_start_time)
            print(f'>> Elapsed time: {elapsed_time}s')
            
            returns.append(episode_reward)
            epi_lens.append(episode_step)

            hf_img, next_image, next_propri = env.reset()
            show_state(hf_img)
            while True:
                _, _, b = get_controller_inputs(joystick)
                if b == 1:
                    break
                else:
                    time.sleep(0.01)
            
            episode_reward = 0
            episode_step = 0
            episode += 1

            print(f'>> -- >> Episode: {episode}')
        
        image = next_image
        propri = next_propri

    buffer.save('reacher_st/hf/hf_fahim')

    # Clean up
    utils.show_learning_curve(dir+'learning curve.png', returns, epi_lens, xtick=2000)

    cv2.destroyAllWindows()
    env.close()

if __name__ == '__main__':
    main()