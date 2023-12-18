import torch
import argparse
import shutil
import time
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADPerformer, SACRADLearner
import relod.utils as utils
from relod.utils import WrappedEnv
from relod.envs.isaac_min_time_reacher.min_reacher_env import CarterMinTimeReacherEnv
from relod.algo.comm import MODE
from relod.logger import Logger
import os

config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent': 50,

    'mlp': [
        [-1, 512],
        [512, 512],
        [512, -1]
    ],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--target_type', default='isaac_min_reacher', type=str)
    parser.add_argument('--image_height', default=90, type=int)
    parser.add_argument('--image_width', default=160, type=int)
    parser.add_argument('--stack_frames', default=3, type=int)
    parser.add_argument('--episode_steps', default=300, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=200000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--init_steps', default=10000, type=int)
    parser.add_argument('--env_steps', default=200000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
    parser.add_argument('--async_buffer', default=True, action='store_true')
    parser.add_argument('--max_updates_per_step', default=1, type=float)
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)
    # critic
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    parser.add_argument('--bootstrap_terminal', default=0, type=int)
    # actor
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    # agent
    parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='l', type=str, help="Modes in ['r', 'l', 'rl'] ")
    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=50000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--xtick', default=3000, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')
    parser.add_argument('--save_path', default='', type=str, help="For saving SAC buffer")
    parser.add_argument('--load_path', default='', type=str, help="Path to SAC buffer file")
    parser.add_argument('--start_step', default=0, type=int)
    parser.add_argument('--start_episode', default=0, type=int)
    args = parser.parse_args()
    return args

def main(seed=-1, env=None):
    args = parse_args()

    if args.mode == 'r':
        mode = MODE.REMOTE_ONLY
    elif args.mode == 'l':
        mode = MODE.LOCAL_ONLY
    elif args.mode == 'rl':
        mode = MODE.REMOTE_LOCAL
    else:
        raise  NotImplementedError()
    
    args.hf_paths = None
    
    if seed != -1:
        args.seed = seed

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if not 'conv' in config:
        image_shape = (0, 0, 0)
    else: 
        image_shape = (3*args.stack_frames, args.image_height, args.image_width)

    args.work_dir += f'/results/{args.target_type}/seed={args.seed}' 

    if os.path.exists(args.work_dir):
        inp = input('The work directory already exists. ' +
                    'Please select one of the following: \n' +  
                    '  1) Press Enter to resume the run.\n' + 
                    '  2) Press X to remove the previous work' + 
                    ' directory and start a new run.\n' + 
                    '  3) Press any other key to exit.\n')
        if inp == 'X' or inp == 'x':
            shutil.rmtree(args.work_dir)
            print('Previous work dir removed.')
        elif inp == '':
            pass
        else:
            exit(0)

    utils.make_dir(args.work_dir)

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    args.model_dir = model_dir
    L = Logger(args.work_dir, use_tb=args.save_tb)

    
    env = WrappedEnv(env,
                     is_min_time=False,
                     episode_max_steps=args.episode_steps, 
                     start_step=args.start_step, 
                     start_episode=args.start_episode)
    
    utils.set_seed_everywhere(args.seed, env)

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.net_params = config
    args.env_action_space = env.action_space

    agent = LocalWrapper(args.episode_steps, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)
    agent.init_performer(SACRADPerformer, args)
    agent.init_learner(SACRADLearner, args, agent.performer)

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)
    
    episode, episode_reward, episode_step, done = 0, 0, 0, True
    image, propri = env.reset()

    # print("------------------------>", args.proprioception_shape, propri.shape)
    agent.send_init_ob((image, propri))
    start_time = time.time()

    returns = []
    epi_lens = []

    update_paused = True

    task_start_time = time.time()

    for step in range(args.env_steps):
        action = agent.sample_action((image, propri))

        (next_image, next_propri), reward, done, info = env.step(action)

        episode_reward += reward
        episode_step += 1

        if not done or 'TimeLimit.truncated' in info:
            mask = 0.0
        else:
            mask = 1.0

        agent.push_sample((image, propri), action, reward, (next_image, next_propri), mask)
        
        if done or 'TimeLimit.truncated' in info:
            elapsed_time = "{:.3f}".format(time.time() - task_start_time)

            L.log('train/duration', time.time() - start_time, step)
            L.log('train/episode_reward', episode_reward, step)
            L.dump(step)
            L.log('train/episode', episode+1, step)
            print(f'>> Elapsed time: {elapsed_time}s')

            returns.append(episode_reward)
            epi_lens.append(episode_step)
            episode_reward = 0
            episode_step = 0
            episode += 1
            start_time = time.time()
            next_image, next_propri = env.reset()
            
            agent.send_init_ob((next_image, next_propri))
            
        if step >= args.init_steps and update_paused:
            update_paused = False
            agent._learner.resume_update()
        
        stat = agent.update_policy(step)
        if stat is not None:
            for k, v in stat.items():
                L.log(k, v, step)
        
        image = next_image
        propri = next_propri

        if args.save_model and (step+1) % args.save_model_freq == 0:
            agent.save_policy_to_file(args.model_dir, step)
            agent.save_buffer()

        if step > 0 and step % args.xtick == 0:
            try:
                utils.show_learning_curve(args.work_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)
            except:
                pass

    if args.save_model:
        agent.save_policy_to_file(args.model_dir, step)
    # Clean up

    utils.show_learning_curve(args.work_dir+'/learning curve.png', returns, epi_lens, xtick=args.xtick)

    agent.close()
    print('Train finished')

if __name__ == '__main__':
    env = CarterMinTimeReacherEnv(scene_path="/home/isaac/.local/share/ov/pkg/isaac_sim-2023.1.0-hotfix.1/reacher_st/relod/relod/envs/isaac_min_time_reacher/arena4.usd",
                           seed=0,
                           image_width=160,
                           image_height=90,
                           img_type='chw',
                           headless=False,
                           min_target_size=0.4)
    for i in range(5):
        env.seed(i)
        main(i, env)
        time.sleep(10)

    env.close()