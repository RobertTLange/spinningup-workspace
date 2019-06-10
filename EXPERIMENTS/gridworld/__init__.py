from gym.envs.registration import register


register(
    id='dense-v0',
    entry_point='gridworld.dense:MazeEnv',
    kwargs=dict(),
    nondeterministic = False,
)
