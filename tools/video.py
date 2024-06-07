import os
import sys

import imageio
import numpy as np


class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, fps=10):
        self.save_dir = root_dir
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []
        self.enabled = True

    def record(self, env):
        if self.enabled:
            frame = env.envs[0].env.render(mode='rgb_array')
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
            self.frames.clear()
    
    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def reset(self):
        self.frames.clear()
