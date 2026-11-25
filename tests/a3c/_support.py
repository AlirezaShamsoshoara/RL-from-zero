"""In-memory process primitives and Gym-compatible stubs, never real physics."""

import queue
import threading

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class StubEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        obs_dim=6,
        act_dim=3,
        episode_length=2,
        reward=-1.0,
        native_reward=None,
        truncated=False,
        fail_reset=False,
        fail_step=False,
    ):
        self.observation_space = spaces.Box(
            -100.0, 100.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(act_dim)
        self.episode_length = episode_length
        self.reward = reward
        self.native_reward = native_reward
        self.truncated = truncated
        self.fail_reset = fail_reset
        self.fail_step = fail_step
        self.reset_seeds = []
        self.close_calls = 0
        self.step_calls = 0
        self.steps = 0
        self.on_step = None

    @property
    def close_called(self):
        return self.close_calls > 0

    def reset(self, *, seed=None, options=None):
        self.reset_seeds.append(seed)
        if self.fail_reset:
            raise RuntimeError("stub reset failed")
        self.steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        if self.on_step is not None:
            self.on_step()
        if self.fail_step:
            raise RuntimeError("stub step failed")
        self.step_calls += 1
        self.steps += 1
        done = self.steps >= self.episode_length
        obs = np.full(self.observation_space.shape, float(self.steps), dtype=np.float32)
        info = {}
        if self.native_reward is not None:
            info["native_reward"] = self.native_reward
            if done:
                info["native_episode_return"] = self.native_reward * self.steps
        return (
            obs,
            self.reward,
            done and not self.truncated,
            done and self.truncated,
            info,
        )

    def close(self):
        self.close_calls += 1

    def render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)


class StubVecEnv:
    """All slots finish together and SAME_STEP returns zero-valued reset states."""

    def __init__(
        self,
        num_envs=2,
        obs_dim=6,
        episode_length=2,
        reward=-1.0,
        native_reward=None,
        truncated=False,
        final_key="final_obs",
    ):
        self.num_envs = num_envs
        self.single_observation_space = spaces.Box(
            -100.0, 100.0, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            -100.0, 100.0, shape=(num_envs, obs_dim), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3] * num_envs)
        self.episode_length = episode_length
        self.reward = reward
        self.native_reward = native_reward
        self.truncated = truncated
        self.final_key = final_key
        self.step_calls = 0
        self.steps = 0
        self.close_calls = 0
        self.reset_seeds = []
        self.on_step = None

    @property
    def close_called(self):
        return self.close_calls > 0

    def reset(self, *, seed=None, options=None):
        self.reset_seeds.append(seed)
        self.steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, actions):
        if self.on_step is not None:
            self.on_step()
        self.step_calls += 1
        self.steps += 1
        done = np.full(self.num_envs, self.steps >= self.episode_length, dtype=bool)
        final_obs = np.full(
            self.observation_space.shape, float(self.steps), dtype=np.float32
        )
        info = {}
        native = self.reward if self.native_reward is None else self.native_reward
        if done.any():
            final_info = {
                "native_reward": np.full(self.num_envs, native),
                "_native_reward": done.copy(),
                "native_episode_return": np.full(self.num_envs, native * self.steps),
                "_native_episode_return": done.copy(),
            }
            info[self.final_key] = final_obs
            info["_" + self.final_key] = done.copy()
            info["final_info"] = final_info
            info["_final_info"] = done.copy()
            self.steps = 0
            obs = np.zeros_like(final_obs)
        else:
            obs = final_obs
            info["native_reward"] = np.full(self.num_envs, native)
        rewards = np.full(self.num_envs, self.reward, dtype=np.float32)
        terminated = done if not self.truncated else np.zeros_like(done)
        truncated = done if self.truncated else np.zeros_like(done)
        return obs, rewards, terminated, truncated, info

    def close(self):
        self.close_calls += 1


class FakeValue:
    def __init__(self, _type="q", value=0):
        self.value = value
        self.lock = threading.RLock()

    def get_lock(self):
        return self.lock


class FakeEvent:
    def __init__(self):
        self.flag = False

    def is_set(self):
        return self.flag

    def set(self):
        self.flag = True


class FakeQueue(queue.Queue):
    def __init__(self):
        super().__init__()
        self.closed = False
        self.joined = False

    def close(self):
        self.closed = True

    def join_thread(self):
        self.joined = True

    def cancel_join_thread(self):
        self.joined = True

    def messages(self):
        result = []
        while True:
            try:
                result.append(self.get_nowait())
            except queue.Empty:
                return result


class SyncFakeProcess:
    """A Process-shaped object that executes the target on the test thread."""

    def __init__(self, target, args, daemon=None, *, context):
        self.target = target
        self.args = args
        self.context = context
        self.daemon = daemon
        self.exitcode = None
        self.started = False
        self.joined = False
        self.terminated = False
        self.closed = False

    def start(self):
        if self.args[0] == self.context.fail_start_worker:
            raise RuntimeError("stub process start failed")
        self.started = True
        if self.args[0] == self.context.silent_failure_worker:
            self.exitcode = 1
            return
        self.target(*self.args)
        self.exitcode = (
            None
            if self.context.alive_after_target
            else self.context.exitcode_after_target
        )

    def is_alive(self):
        return self.started and self.context.alive_after_target and not self.terminated

    def join(self, timeout=None):
        self.joined = True

    def terminate(self):
        self.terminated = True
        self.exitcode = -15

    def kill(self):
        self.terminate()

    def close(self):
        self.closed = True


class FakeContext:
    def __init__(
        self,
        fail_start_worker=None,
        silent_failure_worker=None,
        exitcode_after_target=0,
        alive_after_target=False,
    ):
        self.fail_start_worker = fail_start_worker
        self.silent_failure_worker = silent_failure_worker
        self.exitcode_after_target = exitcode_after_target
        self.alive_after_target = alive_after_target
        self.processes = []
        self.result_queue = None
        self.stop_event = None
        self.values = []
        self.update_lock = None

    def Process(self, target, args, daemon=None):
        process = SyncFakeProcess(target, args, daemon, context=self)
        self.processes.append(process)
        return process

    def Value(self, kind, value):
        result = FakeValue(kind, value)
        self.values.append(result)
        return result

    def Queue(self, *args, **kwargs):
        self.result_queue = FakeQueue()
        return self.result_queue

    def Event(self):
        self.stop_event = FakeEvent()
        return self.stop_event

    def RLock(self):
        self.update_lock = threading.RLock()
        return self.update_lock


class DummyWandb:
    class Run:
        def __init__(self):
            self.finished = False
            self.summary = {}

        def finish(self):
            self.finished = True

    def __init__(self):
        self.logged = []
        self.init_kwargs = None
        self.login_keys = []
        self.run = self.Run()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def log(self, data, step=None):
        self.logged.append((data, step))

    def login(self, key=None):
        self.login_keys.append(key)
