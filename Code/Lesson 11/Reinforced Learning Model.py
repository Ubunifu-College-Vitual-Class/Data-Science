import gym # pip install gym --upgrade

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.wrappers.record_video import RecordVideo
import os

env = gym.make('CartPole-v1', render_mode="rgb_array")
env = RecordVideo(env, './Output',  episode_trigger = lambda episode_number: False)


print(env.action_space)

before_training = os.path.dirname(os.path.realpath(__file__))+"/Output/before_training.mp4"

video = VideoRecorder(env, before_training)

# returns an initial observation
env.reset()

for i in range(200):
  env.render()
  video.capture_frame()

  # env.action_space.sample() produces either 0 (left) or 1 (right).
  response = env.step(env.action_space.sample())
  observation = response[0]
  reward = response[1]
  done = response[2]
  info = response[3]  

  print("step", i, response[0], response[1], response[2], response[3], response[4])
  
  # step (how many times it has cycled through the environment).
  # observation of the environment [x cart position, x cart velocity, pole angle (rad), pole angular velocity]
  # reward achieved by the previous action.
  # done is a boolean. It indicates whether it's time to reset the environment again.
  # info which is diagnostic information useful for debugging.
 
env.reset()
video.close() 
env.close()
