"""
overview:
    train the tx agent using RL
"""

import tensorflow as tf
from keras import layers
from keras import optimizers
import tensorflow_probability as tfp
from utils import load_models_from_dir
from modulate_fn_tf import modulate_psk, modulate_qam, demodulate_psk, demodulate_qam
from models import VQVAE, Actor, Critic
import argparse
import tensorflow_datasets as tfds
from utils import train_preprocessing, val_preprocessing, load_config
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

tf.random.set_seed(42)

FEATURE_DESC = {
    'profile':  tf.io.FixedLenFeature([], tf.string),
    'psnr_all': tf.io.FixedLenFeature([], tf.string),
    'snr':      tf.io.FixedLenFeature([], tf.int64),
}

EPSILON = 0.2
PI = 3.14159265359
PPO_EPOCHS = 4
CLIP_EPS   = 0.2
ENT_COEF   = 0.01
VF_COEF    = 0.5

modulation_schemes = [
    {'modulation_order': 2, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
    {'modulation_order': 4, 'modulate_fn': modulate_psk, 'demodulate_fn': demodulate_psk},
    {'modulation_order': 16, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
    {'modulation_order': 64, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
    {'modulation_order': 256, 'modulate_fn': modulate_qam, 'demodulate_fn': demodulate_qam},
]

def _parse_function(example_proto):
    feature_description = {
        'profile':         tf.io.FixedLenFeature([], tf.string),
        'psnr':            tf.io.FixedLenFeature([], tf.string),
        'tau_rms':         tf.io.FixedLenFeature([], tf.string),
        'fd_rms':          tf.io.FixedLenFeature([], tf.string),
        'snr':             tf.io.FixedLenFeature([], tf.int64),
        'modulation_index':tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    profile = parsed['profile']     
    psnr    = tf.io.parse_tensor(parsed['psnr'], tf.float32)
    tau_rms = tf.io.parse_tensor(parsed['tau_rms'], tf.float32)
    fd_rms  = tf.io.parse_tensor(parsed['fd_rms'], tf.float32)
    snr     = tf.cast(parsed['snr'], tf.int32)
    mod_idx = tf.cast(parsed['modulation_index'], tf.int32)
    return {
        "profile": profile,
        "psnr":    psnr,
        "tau_rms": tau_rms,
        "fd_rms":  fd_rms,
        "snr":     snr,
        "mod_idx": mod_idx
    }

def load_tf_dataset(filename, batch_size=32):
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(
        batch_size,
        padded_shapes={
            "profile": [],  # scalar string
            "psnr":    [],  
            "tau_rms": [],
            "fd_rms":  [],
            "snr":     [],
            "mod_idx": []
        }
    ).prefetch(tf.data.AUTOTUNE)
    return ds

def performance_stats(rewards, psnrs):
    psnrs_1d = [tf.reshape(p, [1]) for p in psnrs]  
    rewards_1d = [tf.reshape(r, [1]) for r in rewards] 

    psnrs_cat = tf.concat(psnrs_1d, axis=0)  
    rewards_cat = tf.concat(rewards_1d, axis=0) 

    psnr_mean = tf.reduce_mean(psnrs_cat)
    reward_mean = tf.reduce_mean(rewards_cat)
    return reward_mean.numpy(), psnr_mean.numpy()

def resize(inputs, target_size):
    return tf.image.resize(inputs, target_size)

def compute_reward(psnr, bits, psnr_t=24.0, lam=0.7):
    p_succ = tf.sigmoid((psnr - psnr_t))
    se = (bits - 1.0) / 7.0
    thr = p_succ * se          # 0~1
    psnr_norm = tf.clip_by_value((psnr - 10.0) / 20.0, 0.0, 1.0)
    return (1- lam) * thr + lam * psnr_norm

@tf.function
def collect_step(images, snr_inputs, psnr_all, tau_rms_all, fd_rms_all, actor, critic):
    inputs_agent = tf.image.resize(images, (8,8))
    logits = actor([inputs_agent, snr_inputs, tau_rms_all, fd_rms_all], training=False)    # [B,5]
    dist   = tfp.distributions.Categorical(logits=logits)
    actions = dist.sample()                                        # [B]
    logp     = dist.log_prob(actions)                             # [B]
    value    = tf.squeeze(critic([inputs_agent, snr_inputs, tau_rms_all, fd_rms_all], training=False), axis=-1)  # [B]
    idx   = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
    psnr  = tf.gather_nd(psnr_all, idx)

    mod_orders = tf.constant([2.,4.,16.,64.,256.], tf.float32)
    bits       = tf.math.log(mod_orders)/tf.math.log(2.0)
    bit    = tf.gather(bits, actions)
    psnr_return = tf.identity(psnr)
    reward = compute_reward(psnr, bit)
    return actions, logp, value, reward, psnr_return

@tf.function
def ppo_update(imgs, snr_, tau_, fd_, acts, lp_old, ret, adv_, actor, critic, actor_opt, critic_opt):    
    with tf.GradientTape(persistent=True) as tape:
        inp = tf.image.resize(imgs, (8,8))
        logits = actor([inp, snr_, tau_, fd_], training=True)
        dist   = tfp.distributions.Categorical(logits=logits)
        logp = dist.log_prob(acts)
        ratio = tf.exp(logp - tf.stop_gradient(lp_old))

        unclipped = ratio * adv_
        clipped   = tf.clip_by_value(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_
        actor_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

        value = tf.squeeze(critic([inp, snr_, tau_, fd_], training=True), axis=-1)
        critic_loss = tf.reduce_mean((ret - value)**2)

        entropy = tf.reduce_mean(dist.entropy())

        total_loss = actor_loss + VF_COEF * critic_loss - ENT_COEF * entropy

    a_grads = tape.gradient(total_loss, actor.trainable_variables)
    c_grads = tape.gradient(total_loss, critic.trainable_variables)
    actor_opt.apply_gradients(zip(a_grads, actor.trainable_variables))
    critic_opt.apply_gradients(zip(c_grads, critic.trainable_variables))

def train_agent(train_ds, actor, critic, actor_opt, critic_opt, args):
    all_states_img, all_states_snr = [], []
    all_actions, all_logp, all_values = [], [], []
    all_rewards = []
    all_psnrs = []
    all_tau_rms = []
    all_fd_rms = []

    # for images, symbol_batch in tqdm(train_ds, desc="Training"):
    for images, symbol_batch in train_ds:
        acts, logp, val, rew, psnr = collect_step(
            images,
            symbol_batch["snr"],
            symbol_batch["psnr_all"],
            symbol_batch["tau_rms_all"],
            symbol_batch["fd_rms_all"],
            actor, critic
        )
        all_states_img.append(images)
        all_states_snr.append(symbol_batch["snr"])
        all_tau_rms.append(symbol_batch["tau_rms_all"])
        all_fd_rms.append(symbol_batch["fd_rms_all"])
        all_actions.append(acts)
        all_logp.append(logp)
        all_values.append(val)
        all_rewards.append(rew)
        all_psnrs.append(psnr)

    S_img   = tf.concat(all_states_img, axis=0)        # [N,H,W,C]
    S_snr   = tf.concat(all_states_snr, axis=0)        # [N,1]
    S_tau   = tf.concat(all_tau_rms, axis=0)           # [N,5]
    S_fd    = tf.concat(all_fd_rms, axis=0)            # [N,5]
    A_old   = tf.concat(all_actions, axis=0)           # [N]
    logp_old= tf.concat(all_logp, axis=0)              # [N]
    V_old   = tf.concat(all_values, axis=0)            # [N]
    R       = tf.concat(all_rewards, axis=0)           # [N]

    adv     = R - V_old
    returns = R  # 1-step return

    dataset = tf.data.Dataset.from_tensor_slices(
        (S_img, S_snr, S_tau, S_fd, A_old, logp_old, returns, adv)
    ).shuffle(1024).batch(args.batch_size)

    for _ in range(PPO_EPOCHS):
        for (imgs, snr_, tau_, fd_, acts, lp_old, ret, adv_) in dataset:
            ppo_update(
                imgs, snr_, tau_, fd_, acts, lp_old, ret, adv_,
                actor, critic, actor_opt, critic_opt
            )
    return tf.reduce_mean(R), tf.reduce_mean(all_psnrs)

@tf.function
def test_step(images, snr_inputs, psnr_all, tau_rms_all, fd_rms_all, actor):
    inputs_agent =  tf.identity(images)
    inputs_agent = resize(inputs_agent, (8, 8))

    logits = actor([inputs_agent, snr_inputs, tau_rms_all, fd_rms_all], training=False)              # [B,5]
    dist   = tfp.distributions.Categorical(logits=logits)
    actions = dist.sample()                                         # [B]

    idx       = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
    psnr_chosen = tf.gather_nd(psnr_all, idx)                      # [B]
    psnr_return = tf.identity(psnr_chosen)

    mod_orders = tf.constant([2.,4.,16.,64.,256.], dtype=tf.float32)
    bits       = tf.math.log(mod_orders) / tf.math.log(2.0)         # [5]
    bits_chosen  = tf.gather(bits, actions)                         # [B]

    reward_actor = compute_reward(psnr_chosen, bits_chosen)
    return tf.reduce_mean(reward_actor), tf.reduce_mean(psnr_return)

def test_agent(val_ds, actor):
    psnr_list, rewards = [], []
    rewards = []

    for images, symbol_batch in val_ds:
        reward, psnr = test_step(
            images,
            symbol_batch["snr"],
            symbol_batch["psnr_all"],
            symbol_batch["tau_rms_all"],
            symbol_batch["fd_rms_all"],
            actor
        )

        rewards.append(reward)
        psnr_list.append(psnr)
    return performance_stats(rewards, psnr_list)

def plot_agent_action(val_ds, actor, save_img_dir, epoch, profile, snr_values=None):
    if snr_values is None:
        snr_values = list(range(0, 60, 2))
    
    for images, symbol_batch in val_ds.take(1):
        ref_image    = images[0:1]
        tau_rms_take = symbol_batch["tau_rms_all"][0:1]
        fd_rms_take  = symbol_batch["fd_rms_all"][0:1]
        break

    num_snr = len(snr_values)
    imgs         = tf.tile(ref_image,    [num_snr,1,1,1])
    snrs         = tf.constant(snr_values, dtype=tf.float32)[:,None]
    tau_tiled    = tf.tile(tau_rms_take, [num_snr,1])
    fd_tiled     = tf.tile(fd_rms_take,  [num_snr,1])
    resize_imgs  = resize(imgs, (8,8))
    
    logits = actor([resize_imgs, snrs, tau_tiled, fd_tiled], training=False)
    probs  = tf.nn.softmax(logits, axis=-1).numpy()
    best_actions = probs.argmax(axis=1)
    action_labels = ["BPSK","QPSK","16QAM","64QAM","256QAM"]

    out_dir = os.path.join(save_img_dir, profile)
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(snr_values, best_actions, marker='o')
    plt.xlabel("SNR (dB)"); plt.ylabel("Selected Modulation")
    plt.title(f"{profile}: Action vs SNR (epoch {epoch+1})")
    plt.yticks(range(len(action_labels)), action_labels)
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"action_choice_epoch{epoch+1}.png"))
    plt.close()

    plt.figure()
    for i, label in enumerate(action_labels):
        plt.plot(snr_values, probs[:,i], marker='o', label=label)
    plt.xlabel("SNR (dB)"); plt.ylabel("Action Probability")
    plt.title(f"{profile}: Action Prob vs SNR (epoch {epoch+1})")
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"action_dist_epoch{epoch+1}.png"))
    plt.close()

def dump_grouped_dataset_to_tfrecord(grouped_ds: tf.data.Dataset, output_path: str):
    writer = tf.io.TFRecordWriter(output_path)
    for elem in grouped_ds.unbatch().as_numpy_iterator():
        feats = {
            'psnr_all':    tf.train.Feature(bytes_list=tf.train.BytesList(
                                value=[tf.io.serialize_tensor(elem['psnr_all']).numpy()])),
            'tau_rms_all': tf.train.Feature(bytes_list=tf.train.BytesList(
                                value=[tf.io.serialize_tensor(elem['tau_rms_all']).numpy()])),
            'fd_rms_all':  tf.train.Feature(bytes_list=tf.train.BytesList(
                                value=[tf.io.serialize_tensor(elem['fd_rms_all']).numpy()])),
            'snr':         tf.train.Feature(int64_list=tf.train.Int64List(
                                value=[int(elem['snr'][0])]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feats))
        writer.write(example.SerializeToString())
    writer.close()

def load_grouped_tfrecord(filename: str, batch_size: int = 32) -> tf.data.Dataset:
    feature_description = {
        'psnr_all':    tf.io.FixedLenFeature([], tf.string),
        'tau_rms_all': tf.io.FixedLenFeature([], tf.string),
        'fd_rms_all':  tf.io.FixedLenFeature([], tf.string),
        'snr':         tf.io.FixedLenFeature([], tf.int64),
    }
    def _parse_fn(proto):
        parsed     = tf.io.parse_single_example(proto, feature_description)
        psnr_all   = tf.io.parse_tensor(parsed['psnr_all'],    out_type=tf.float32)
        tau_rms_all= tf.io.parse_tensor(parsed['tau_rms_all'], out_type=tf.float32)
        fd_rms_all = tf.io.parse_tensor(parsed['fd_rms_all'],  out_type=tf.float32)
        snr        = tf.cast(parsed['snr'], tf.float32)
        return {
            'psnr_all':    psnr_all,
            'tau_rms_all': tau_rms_all,
            'fd_rms_all':  fd_rms_all,
            'snr':         tf.expand_dims(snr, -1)
        }

    ds = tf.data.TFRecordDataset(filename)
    ds = ds.map(_parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def main(args):
    max_epoch = args.max_epoch
    vqvae_model_dir = args.vqvae_model_dir
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    batch_size = args.batch_size
    img_size = args.img_size
    save_img_dir = args.save_img_dir

    if args.dataset_name == 'eurosat':
        train_ds_raw = tfds.load(args.dataset_name, split="train[:80%]", with_info=False, shuffle_files=False)
        val_ds_raw = tfds.load(args.dataset_name, split="train[80%:90%]", with_info=False, shuffle_files=False)
    else:
        train_ds_raw = tfds.load(args.dataset_name, split="train[:90%]", with_info=False, shuffle_files=False)
        val_ds_raw = tfds.load(args.dataset_name, split="train[90%:]", with_info=False, shuffle_files=False)

    train_dataset = (
        train_ds_raw
        .map(lambda x: train_preprocessing(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(batch_size * 2, seed=42, reshuffle_each_iteration=False)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (val_ds_raw
            .map(lambda x: val_preprocessing(x, img_size), num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(batch_size * 2, seed=42, reshuffle_each_iteration=False)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
            )

    best_avg_val_reward = -1e3

    ACTION_DIM = 5  

    actor = Actor(ACTION_DIM)
    critic = Critic()

    actor_optimizer = optimizers.Adam(learning_rate=actor_lr)
    critic_optimizer = optimizers.Adam(learning_rate=critic_lr)

    vqvae_model = VQVAE(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        num_modulations=args.num_modulations,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
        n_res_block=args.n_res_block,
        img_size=img_size,
    )
    vqvae_model = load_models_from_dir(vqvae_model_dir, vqvae_model)

    symbol_ds = load_tf_dataset(
        f"doppler_data/{args.dataset_name}/doppler_psnr_train_dataset.tfrecord",
        batch_size=batch_size
    )

    symbol_ds_val = load_tf_dataset(
        f"doppler_data/{args.dataset_name}/doppler_psnr_val_dataset.tfrecord",
        batch_size=batch_size
    )

    ds_unbatched = symbol_ds.unbatch().prefetch(tf.data.AUTOTUNE)
    ds_unbatched_val = symbol_ds_val.unbatch().prefetch(tf.data.AUTOTUNE)

    profile_list = ['NTN-TDL-A','NTN-TDL-B','NTN-TDL-C','NTN-TDL-D']
    symbol_grouped_by_profile = {}
    symbol_grouped_by_profile_val = {}

    for prof in profile_list:
        prof_t = tf.constant(prof)
        ds_prof = ds_unbatched.filter(lambda ex, p=prof_t: tf.equal(ex['profile'], p))

        ds_mods = []
        for i in range(ACTION_DIM):
            ds_i = (
                ds_prof
                .filter(lambda ex, i=i: tf.equal(ex['mod_idx'], i))
                .map(lambda ex: (
                    ex['psnr'],    # [T]
                    ex['tau_rms'], # [T]
                    ex['fd_rms'],  # [T]
                    ex['snr']      # scalar
                ))
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE)
            )
            ds_mods.append(ds_i)

        symbol_grouped_by_profile[prof] = tf.data.Dataset.zip(tuple(ds_mods)).map(
            lambda *mod_exs: {
                "psnr_all":    tf.stack([mod_exs[j][0] for j in range(ACTION_DIM)], axis=1),  # [B,5,T]
                "tau_rms_all": tf.stack([mod_exs[j][1] for j in range(ACTION_DIM)], axis=1),  # [B,5,T]
                "fd_rms_all":  tf.stack([mod_exs[j][2] for j in range(ACTION_DIM)], axis=1),  # [B,5,T]
                "snr":         tf.cast(mod_exs[0][3], tf.float32)[...,None]                   # [B,1]
            }
        )

        ds_prof_val = ds_unbatched_val.filter(lambda ex, p=prof_t: tf.equal(ex['profile'], p))
        ds_mods_val = []
        for i in range(ACTION_DIM):
            ds_i = (
                ds_prof_val
                .filter(lambda ex, i=i: tf.equal(ex['mod_idx'], i))
                .map(lambda ex: (
                    ex['psnr'],
                    ex['tau_rms'],
                    ex['fd_rms'],
                    ex['snr']
                ))
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE)
            )
            ds_mods_val.append(ds_i)

        symbol_grouped_by_profile_val[prof] = tf.data.Dataset.zip(tuple(ds_mods_val)).map(
            lambda *mod_exs: {
                "psnr_all":    tf.stack([mod_exs[j][0] for j in range(ACTION_DIM)], axis=1),
                "tau_rms_all": tf.stack([mod_exs[j][1] for j in range(ACTION_DIM)], axis=1),
                "fd_rms_all":  tf.stack([mod_exs[j][2] for j in range(ACTION_DIM)], axis=1),
                "snr":         tf.cast(mod_exs[0][3], tf.float32)[...,None]
            }
        )

    os.makedirs(f"split_train/{args.dataset_name}", exist_ok=True)
    os.makedirs(f"split_val/{args.dataset_name}", exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    for prof in profile_list:
        out_train = f"split_train/{args.dataset_name}/doppler_{prof}.tfrecord"
        if not os.path.exists(out_train):
            print(f"Dumping {out_train} …")
            dump_grouped_dataset_to_tfrecord(symbol_grouped_by_profile[prof], out_train)
        else:
            print(f"{out_train} already exists, skipping.")

        out_val = f"split_val/{args.dataset_name}/doppler_{prof}.tfrecord"
        if not os.path.exists(out_val):
            print(f"Dumping {out_val} …")
            dump_grouped_dataset_to_tfrecord(symbol_grouped_by_profile_val[prof], out_val)
        else:
            print(f"{out_val} already exists, skipping.")

    for epoch in range(max_epoch):
        train_rewards = []
        train_psnrs = []
        for prof in profile_list:
            train_file = f"split_train/{args.dataset_name}/doppler_{prof}.tfrecord"
            ds_train_symbols = load_grouped_tfrecord(train_file, batch_size=args.batch_size)
            train_ds_prof   = tf.data.Dataset.zip((train_dataset, ds_train_symbols))

            r_train, psnr_train = train_agent(
                train_ds_prof, actor, critic, actor_optimizer, critic_optimizer, args
            )
            train_rewards.append(r_train)
            train_psnrs.append(psnr_train)

        avg_train_reward = sum(train_rewards) / len(train_rewards)
        avg_train_psnr = sum(train_psnrs) / len(train_psnrs)

        val_rewards = []
        val_psnrs = []
        for prof in profile_list:
            val_file = f"split_val/{args.dataset_name}/doppler_{prof}.tfrecord"
            ds_val_symbols = load_grouped_tfrecord(val_file, batch_size=args.batch_size)
            val_ds_prof    = tf.data.Dataset.zip((val_dataset, ds_val_symbols))

            r_val, psnr_val = test_agent(val_ds_prof, actor)
            val_rewards.append(r_val)
            val_psnrs.append(psnr_val)

        avg_val_reward = sum(val_rewards) / len(val_rewards)
        avg_val_psnr = sum(val_psnrs) / len(val_psnrs)
        print(f"Epoch {epoch+1} | Train Reward: {avg_train_reward:.4f}, Train PSNR: {avg_train_psnr:.3f}, Avg Val Reward: {avg_val_reward:.4f}, Avg Val PSNR: {avg_val_psnr:.3f}")

        if avg_val_reward > best_avg_val_reward:
            best_avg_val_reward = avg_val_reward
            os.makedirs(args.save_model_dir, exist_ok=True)
            actor.save_weights(os.path.join(args.save_model_dir, 'tx_actor.h5'))
            critic.save_weights(os.path.join(args.save_model_dir, 'tx_critic.h5'))
            print("  ▶ New best avg val reward; model saved.")

        if (epoch + 1) % 10 == 0:
            for prof in profile_list:
                val_file = f"split_val/{args.dataset_name}/doppler_{prof}.tfrecord"
                ds_val_symbols = load_grouped_tfrecord(val_file, batch_size=args.batch_size)
                val_ds_prof    = tf.data.Dataset.zip((val_dataset, ds_val_symbols))
                plot_agent_action(val_ds_prof, actor, save_img_dir, epoch, prof)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='cifar10', help="Name of the dataset to use", choices=['cifar10', 'eurosat'])
    args = parser.parse_args()

    parser.add_argument("--config", type=str, default=f"config/{args.dataset_name}/model_config.yaml", help="Path to the config file")
    args = parser.parse_args()
    config = load_config(args.config)

    parser.add_argument("--img_size", type=int, default=config.get('img_size', 32))
    parser.add_argument("--max_epoch", type=int, default=config.get('rl_max_epoch', 200))
    parser.add_argument("--num_modulations", type=int, default=config.get('num_modulations', 5))
    parser.add_argument("--num_embeddings", type=int, default=config.get('num_embeddings', 512))
    parser.add_argument("--commitment_cost", type=float, default=config.get('commitment_cost', 0.25))
    parser.add_argument("--decay", type=float, default=config.get('decay', 0.99))
    parser.add_argument("--actor_lr", type=float, default=config.get('actor_lr', 1e-4))
    parser.add_argument("--critic_lr", type=float, default=config.get('critic_lr', 5e-4))

    parser.add_argument("--batch_size", type=int, default=config.get('batch_size', 128))
    parser.add_argument("--embedding_dim", type=int, default=config.get('embedding_dim', 32))
    parser.add_argument("--n_res_block", type=int, default=config.get('n_res_block', 2))
    parser.add_argument("--save_model_dir", type=str, default=config.get('rl_rx_agent_model_dir', f'./rl_model/{args.dataset_name}'))
    parser.add_argument("--save_img_dir", type=str, default=config.get('rl_rx_agent_img_dir', f'./rl_images/{args.dataset_name}'))
    parser.add_argument("--vqvae_model_dir", type=str, default=config.get('pretrain_vqvae_model_dir', f'./vqvae_model/{args.dataset_name}'))

    args = parser.parse_args()

    print(args)
    main(args)
