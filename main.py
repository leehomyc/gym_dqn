"""Deep Q learning for gym frozen lake."""
import logging

import gym
import numpy as np
import tensorflow as tf

del logging.getLogger().handlers[:]
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    level=logging.INFO,
)


# -------------------------------------------------------------------

logger = logging.getLogger(__name__)

ENV = gym.make('FrozenLake-v0')

_OBS_SPACE = ENV.observation_space.n
_ACT_SPACE = ENV.action_space.n

# Future reward discount coefficient.
_DISC_COEFF = .99
_NUM_EPISODES = 200
_NUM_STEPS = 99

# -------------------------------------------------------------------


def model_setup():
    """Set up the input and output.

    Returns:
        inputs: A one-hot tensor as the index of the state.
        q_out: A tensor as the predicted reward.
        predict: A tensor as the optimal action.
        next_q: A tensor as the target reward.
    """
    inputs = tf.placeholder(shape=[1, _OBS_SPACE], dtype=tf.float32)
    weight = tf.Variable(tf.random_uniform([_OBS_SPACE, _ACT_SPACE], 0, 0.01))
    q_out = tf.matmul(inputs, weight)
    predict = tf.argmax(q_out, 1)
    next_q = tf.placeholder(shape=[1, _ACT_SPACE], dtype=tf.float32)
    return inputs, predict, next_q, q_out


def loss_setup(next_q, q_out):
    """Set up the loss.

    Returns:
        A tensor as the L2 loss of predicted reward and target reward.
    """
    return tf.reduce_sum(tf.square(next_q - q_out))


def train():
    """The main training function."""
    tf.reset_default_graph()
    inputs, predict, next_q, q_out = model_setup()
    loss = loss_setup(next_q, q_out)

    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    update_model = trainer.minimize(loss)

    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())

    # create lists to contain total rewards.
    reward_list = []

    explore_thresh = .1
    reward_all = 0
    with tf.Session() as sess:
        sess.run(init)
        for i in range(_NUM_EPISODES):
            s = ENV.reset()
            for j in range(_NUM_STEPS):
                pred, q_val = sess.run([predict, q_out], feed_dict={
                    inputs: np.identity(_OBS_SPACE)[s:s + 1]})
                if np.random.rand(1) < explore_thresh:
                    pred[0] = ENV.action_space.sample()

                new_s, action_reward, final_state, _ = ENV.step(pred[0])

                new_q = sess.run(q_out, feed_dict={
                    inputs: np.identity(_OBS_SPACE)[new_s:new_s + 1]})

                target_q = new_q
                target_q[0, pred[0]] = action_reward + \
                    _DISC_COEFF * np.max(new_q)

                sess.run([update_model], feed_dict={inputs: np.identity(
                    _OBS_SPACE)[s:s + 1], next_q: target_q})
                reward_all += action_reward
                s = new_s
                if final_state is True:
                    # Reduce chance of random action as we train the model.
                    explore_thresh = 1. / ((i / 50) + 10)
                    break
            reward_list.append(reward_all)

    logger.info("percent of successful episodes: {}.".format(
                str(sum(reward_list) / _NUM_EPISODES)))

if __name__ == '__main__':
    train()
