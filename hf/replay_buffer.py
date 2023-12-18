import numpy as np
import collections
import threading
import time
import os
import pickle
from multiprocessing import shared_memory, Process, Queue, Lock


Batch = collections.namedtuple(
    'Batch', ['images', 'proprioceptions', 'actions', 'rewards',
              'masks', 'next_images', 'next_proprioceptions'])


class RadReplayBuffer():
    """Buffer to store environment transitions."""

    def __init__(self, image_shape, proprioception_shape, action_shape,
                 capacity, batch_size, init_buffers=True, load_path=''):

        self._image_shape = image_shape
        self._proprioception_shape = proprioception_shape
        self._action_shape = action_shape
        self._capacity = capacity
        self._batch_size = batch_size

        self._idx = 0
        self._full = False
        self._count = 0
        self._steps = 0
        self._lock = Lock()

        self._ignore_image = True
        self._ignore_propri = True

        if image_shape[-1] != 0:
            self._ignore_image = False

        if proprioception_shape[-1] != 0:
            self._ignore_propri = False

        if load_path:
            self._load_path = load_path
        else:
            self._load_path = ''

        if init_buffers:
            self._init_buffers()

    def _init_buffers(self):
        if self._load_path:
            self._load()
        else:
            if not self._ignore_image:
                self._images = np.empty(
                    (self._capacity, *self._image_shape), dtype=np.uint8)
                self._next_images = np.empty(
                    (self._capacity, *self._image_shape), dtype=np.uint8)

            if not self._ignore_propri:
                self._propris = np.empty(
                    (self._capacity, *self._proprioception_shape), 
                    dtype=np.float32)
                self._next_propris = np.empty(
                    (self._capacity, *self._proprioception_shape), 
                    dtype=np.float32)

            self._actions = np.empty(
                (self._capacity, *self._action_shape), dtype=np.float32)
            
            self._rewards = np.empty((self._capacity), dtype=np.float32)
            self._masks = np.empty((self._capacity), dtype=np.float32)

    def add(self, image, propri, action, reward, next_image, next_propri, mask):
        if not self._ignore_image:
            self._images[self._idx] = image
            self._next_images[self._idx] = next_image
        if not self._ignore_propri:
            self._propris[self._idx] = propri
            self._next_propris[self._idx] = next_propri
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._masks[self._idx] = mask

        self._idx = (self._idx + 1) % self._capacity
        self._full = self._full or self._idx == 0
        self._count = self._capacity if self._full else self._idx
        self._steps += 1

    def sample(self):
        idxs = np.random.randint(0, self._count,
                                 size=min(self._count, self._batch_size))
        
        if self._ignore_image:
            images = None
            next_images = None
        else:
            images = self._images[idxs]
            next_images = self._next_images[idxs]

        if self._ignore_propri:
            propris = None
            next_propris = None
        else:
            propris = self._propris[idxs]
            next_propris = self._next_propris[idxs]

        actions = self._actions[idxs]
        rewards = self._rewards[idxs]
        masks = self._masks[idxs]

        return Batch(images=images, proprioceptions=propris,
                     actions=actions, rewards=rewards, masks=masks,
                     next_images=next_images, next_proprioceptions=next_propris)


    def save(self, save_path):
        tic = time.time()
        print(f'Saving the replay buffer in {save_path}..')
        with self._lock:
            data = {
                'count': self._count,
                'idx': self._idx,
                'full': self._full,
                'steps': self._steps
            }

            with open(os.path.join(save_path, "buffer_data.pkl"),
                      "wb") as handle:
                pickle.dump(data, handle, protocol=4)

            if not self._ignore_image:
                np.save(os.path.join(save_path, "images.npy"), self._images)
                np.save(os.path.join(save_path, "next_images.npy"),
                        self._next_images)

            if not self._ignore_propri:
                np.save(os.path.join(save_path, "propris.npy"), self._propris)
                np.save(os.path.join(save_path, "next_propris.npy"),
                        self._next_propris)

            np.save(os.path.join(save_path, "actions.npy"), self._actions)
            np.save(os.path.join(save_path, "rewards.npy"), self._rewards)
            np.save(os.path.join(save_path, "masks.npy"), self._masks)

        print("Saved the buffer locally,", end=' ')
        print("took: {:.3f}s.".format(time.time() - tic))

    def _load(self):
        tic = time.time()
        print("Loading buffer")

        data = pickle.load(open(os.path.join(self._load_path,
                                             "buffer_data.pkl"), "rb"))
        self._count = data['count']
        self._idx = data['idx']
        self._full = data['full']
        self._steps = data['steps']

        if not self._ignore_image:
            self._images = np.load(os.path.join(self._load_path, "images.npy"))
            self._next_images = np.load(os.path.join(self._load_path,
                                                     "next_images.npy"))

        if not self._ignore_propri:
            self._propris = np.load(os.path.join(self._load_path,
                                                 "propris.npy"))
            self._next_propris = np.load(os.path.join(self._load_path,
                                                      "next_propris.npy"))

        self._actions = np.load(os.path.join(self._load_path, "actions.npy"))
        self._rewards = np.load(os.path.join(self._load_path, "rewards.npy"))
        self._masks = np.load(os.path.join(self._load_path, "masks.npy"))

        print("Loaded the buffer from: {}".format(self._load_path), end=' ')
        print("Took: {:.3f}s".format(time.time() - tic))

