import tensorflow as tf
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
from shutil import copyfile, rmtree


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir, backup=False):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)
        self._log_dir = log_dir
        self._backup = backup

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_scalar_avg(self, tag, value_list, avglen, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value_list[-1])])
        self.writer.add_summary(summary, step)

        avg_value = 0
        if len(value_list) > avglen:
            avg_value = sum(value_list[-avglen:])/avglen
        else:
            avg_value = sum(value_list)/len(value_list)

        summary = tf.Summary(value=[tf.Summary.Value(tag="Avg_{}".format(tag), simple_value=avg_value)])
        self.writer.add_summary(summary, step)

    def log_scalar_rl(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag+"_num_episodes", simple_value=value)])
        self.writer.add_summary(summary, step[0])

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag+"_num_steps", simple_value=value)])
        self.writer.add_summary(summary, step[1])

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag+"_num_updates", simple_value=value)])
        self.writer.add_summary(summary, step[2])


    def log_scalar_rl_w_average(self, tag, value_list, avglen, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag+"_num_episodes", simple_value=value_list[-1])])
        self.writer.add_summary(summary, step[0])

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag+"_num_steps", simple_value=value_list[-1])])
        self.writer.add_summary(summary, step[1])

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag+"_num_updates", simple_value=value_list[-1])])
        self.writer.add_summary(summary, step[2])

        avg_value = 0
        if len(value_list) > avglen:
            avg_value = float(sum(value_list[-avglen:]))/avglen
        else:
            avg_value = float(sum(value_list))/len(value_list)

        summary = tf.Summary(value=[tf.Summary.Value(tag="avg_"+tag+"_num_episodes", simple_value=avg_value)])
        self.writer.add_summary(summary, step[0])

        summary = tf.Summary(value=[tf.Summary.Value(tag="avg_"+tag+"_num_steps", simple_value=avg_value)])
        self.writer.add_summary(summary, step[1])

        summary = tf.Summary(value=[tf.Summary.Value(tag="avg_"+tag+"_num_updates", simple_value=avg_value)])
        self.writer.add_summary(summary, step[2])

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def _append_to(self, tlist, tr, val):
        tlist[0].append(val)
        tlist[1].append([tr.episodes_done, tr.global_steps_done, tr.iterations_done])

    def track_a2c_training_metrics(self, episode_dones, rewards):
        for env_id, episode_done in enumerate(episode_dones):
            self._training_metrics["episode_lens"][env_id] +=1.0
            self._training_metrics["rewards"][env_id] += rewards[env_id]

            if episode_done:
                self._tr.episodes_done += 1
                if env_id==0:
                      self.log_a2c_training_metrics()
                self._training_metrics["episode_lens"][env_id] = 0.0
                self._training_metrics["rewards"][env_id] = 0.0

    def reset_a2c_training_metrics(self, num_envs, tr, avglen):
        self._tr = tr
        self._avglen = avglen
        self._training_metrics = {}
        for tag in ["episode_lens", "rewards"]:
            self._training_metrics[tag] = [0]*num_envs

    def log_a2c_training_metrics(self):
        self._append_to(self._tr.train_reward, self._tr, self._training_metrics["rewards"][0])
        self._append_to(self._tr.train_episode_len, self._tr, self._training_metrics["episode_lens"][0])

        self.log_scalar_rl("train_rewards", self._tr.train_reward[0], self._avglen, [self._tr.episodes_done, self._tr.global_steps_done, self._tr.iterations_done])
        self.log_scalar_rl("train_episode_len", self._tr.train_episode_len[0], self._avglen, [self._tr.episodes_done, self._tr.global_steps_done, self._tr.iterations_done])

    def save(self, dir_name):
        if self._backup:
            create_folder(dir_name)
            rmtree(dir_name)
            create_folder(dir_name)

            tfpath = self._log_dir
            onlyfiles = [f for f in listdir(tfpath) if isfile(join(tfpath, f))]

            for file in onlyfiles:
                copyfile(join(tfpath,file),join(dir_name, file))

    def load(self, dir_name):
        if self._backup:
            rmtree(self._log_dir)
            create_folder(self._log_dir)

            onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]

            for file in onlyfiles:
                copyfile(join(dir_name, file), join(self._log_dir, file))

    def force_restart(self):
        create_folder(self._log_dir)
        rmtree(self._log_dir)
        self.writer = tf.summary.FileWriter(self._log_dir)
        #import pdb; pdb.set_trace()
