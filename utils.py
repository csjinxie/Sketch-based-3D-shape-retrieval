import tensorflow as tf



def get_checkpoint(path):
    print("[*] Reading checkpoint ...")
    ckpt = tf.train.get_checkpoint_state(path)

    if ckpt and ckpt.model_checkpoint_path:
        return ckpt.model_checkpoint_path
    else:
        return None
