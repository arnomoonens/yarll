from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_addons as tfa

from yarll.agents.agent import Agent
from yarll.environment.environment import Environment
from yarll.agents.env_runner import EnvRunner
from yarll.misc.utils import flatten_list
from yarll.memory.experiences_memory import ExperiencesMemory

class FittedQIteration(Agent):
    def __init__(self,
                 env: Environment,
                 monitor_path: str,
                 checkpoint_path: Optional[Union[Path, str]] = None,
                 **usercfg) -> None:
        super().__init__()

        self.env = env
        self.monitor_path = Path(monitor_path)

        self.config.update(
            gamma=0.99,
            alpha=0.5,
            epsilon=0.1,
            epsilon_decay=0.01,
            learning_rate=1e-3,
            n_hidden_layers=2,
            n_hidden_units=64,
            n_iterations=10000,
            batch_update="trajectories",
            trajectories_per_batch=1,
            n_epochs=5,
            normalize_states=False,
            checkpoints=True,
            checkpoint_every_iterations=10,
            checkpoints_max_to_keep=None,
            save_model=True,
        )
        self.config.update(usercfg)

        self.n_actions = self.env.action_space.n
        self.epsilon = self.config["epsilon"]

        self.q_network = self.make_q_network()
        self.q_network.compile(optimizer=tfa.optimizers.RectifiedAdam(self.config["learning_rate"]),
                               loss="mse")
        self.q_network.build((None, env.observation_space.shape[0] + self.n_actions))
        self.ckpt = tf.train.Checkpoint(net=self.q_network)

        if self.config["checkpoints"]:
            checkpoint_directory = self.monitor_path / "checkpoints"
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                           checkpoint_directory,
                                                           max_to_keep=self.config["checkpoints_max_to_keep"])

        if checkpoint_path is not None:
            self.ckpt.restore(str(checkpoint_path)).expect_partial() # With .assert_consumed() it gives errors...

        self.writer = tf.summary.create_file_writer(str(self.monitor_path))
        self.tensorboard_cbk = tf.keras.callbacks.TensorBoard(log_dir=self.monitor_path)
        self.env_runner = EnvRunner(self.env,
                                    self,
                                    self.config,
                                    scale_states=self.config["normalize_states"])

    def make_q_network(self):
        model = tf.keras.Sequential()
        for _ in range(self.config["n_hidden_layers"]):
            model.add(Dense(self.config["n_hidden_units"], activation="relu"))
        model.add(Dense(1))
        return model

    @tf.function
    def action(self, state):
        tiled_state = tf.tile([state], [self.n_actions, 1])
        actions_onehot = tf.one_hot(tf.range(self.n_actions), depth=self.n_actions, dtype=tf.float32)
        inp = tf.concat([tiled_state, actions_onehot], axis=1)
        q_values = self.q_network(inp)
        return tf.argmax(q_values)[0]

    def choose_action(self, state, *rest) -> dict:
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = self.action(state.astype(np.float32)).numpy()
        return {"action": action}

    @staticmethod
    def get_processed_trajectories(trajectories: List[ExperiencesMemory]):
        states = tf.convert_to_tensor(flatten_list([t.states for t in trajectories]), dtype=tf.float32)
        actions = tf.convert_to_tensor(flatten_list([t.actions for t in trajectories]), dtype=tf.int32)
        rewards = tf.convert_to_tensor(flatten_list([t.rewards for t in trajectories]), dtype=tf.float32)
        next_states = tf.convert_to_tensor(flatten_list([t.next_states for t in trajectories]), dtype=tf.float32)
        terminals = tf.convert_to_tensor(flatten_list([t.terminals for t in trajectories]), dtype=tf.float32)
        return states, actions, rewards, next_states, terminals

    @tf.function(experimental_relax_shapes=True)
    def calculate_target_q(self, rewards: tf.Tensor, next_states: tf.Tensor, terminals: tf.Tensor):
        n_states = len(rewards)
        # For every state, make a sample with the one-hot of every action concatenated to it
        oh = tf.eye(self.n_actions)
        repeated_oh = tf.repeat(oh, n_states, axis=0)
        repeated_next_states = tf.tile(next_states, [self.n_actions, 1])
        next_states_ohs = tf.concat([repeated_next_states, repeated_oh], axis=1)
        # Predict q values and calculate max for every state
        q_next_state = self.q_network(next_states_ohs)
        max_q = tf.reduce_max(tf.reshape(q_next_state, (self.n_actions, n_states)), axis=0)

        return rewards + self.config["gamma"] * max_q * (1 - terminals)

    def learn(self):
        with self.writer.as_default():
            for i in range(self.config["n_iterations"]):
                trajs = self.env_runner.get_trajectories()
                states, actions, rewards, next_states, terminals = self.get_processed_trajectories(trajs)
                target_q = self.calculate_target_q(rewards, next_states, terminals)
                actions_oh = tf.one_hot(actions, depth=self.n_actions, dtype=tf.float32)
                states_actions_oh = tf.concat([states, actions_oh], axis=1)
                history = self.q_network.fit(states_actions_oh,
                                             target_q,
                                             epochs=self.config["n_epochs"],
                                             verbose=0)
                tf.summary.scalar("model/loss/mean",
                                  np.average(history.history["loss"]),
                                  step=self.env_runner.total_steps)
                tf.summary.scalar("model/epsilon",
                                  self.epsilon,
                                  step=self.env_runner.total_steps)
                self.epsilon = self.epsilon * (1.0 - self.config["epsilon_decay"])
                if self.config["checkpoints"] and (i % self.config["checkpoint_every_iterations"]) == 0:
                    self.ckpt_manager.save(i)
