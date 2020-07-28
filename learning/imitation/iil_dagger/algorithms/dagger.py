import math
from .iil_learning import InteractiveImitationLearning
import numpy as np

class DAgger(InteractiveImitationLearning):
    """
    DAgger algorithm to mix policies between learner and expert 
    Ross, Stéphane, Geoffrey Gordon, and Drew Bagnell. "A reduction of imitation learning and structured prediction to no-regret online learning." Proceedings of the fourteenth international conference on artificial intelligence and statistics. 2011.
    ...
    Methods
    -------
    _mix
        used to return a policy teacher / expert based on random choice and safety checks
    """

    def __init__(self, env, teacher, learner, horizon, episodes, alpha=0.5, test=False):
        InteractiveImitationLearning.__init__(self, env, teacher, learner, horizon, episodes, test)
        # expert decay
        self.p = alpha
        self.alpha = self.p

        # thresholds used to give control back to learner once the teacher converges
        self.convergence_distance = 0.05
        self.convergence_angle = np.pi / 18 

        # threshold on angle and distance from the lane when using the model to avoid going off track and env reset within an episode
        self.angle_limit = np.pi / 8
        self.distance_limit = 0.12

    def _mix(self):
        control_policy = np.random.choice(
            a=[self.teacher, self.learner],
            p=[self.alpha, 1. - self.alpha]
        )
        if self._found_obstacle:
            return self.teacher
        try:
            lp = self.environment.get_lane_pos2(self.environment.cur_pos, self.environment.cur_angle)
        except :
            return control_policy
        if self.active_policy:
            # keep using tecaher untill duckiebot converges back on track
            if not(abs(lp.dist) < self.convergence_distance and abs(lp.angle_rad)< self.convergence_angle):
                return self.teacher
        else:
            # in case we are using our learner and it started to diverge a lot we need to give 
            # control back to the expert 
            if abs(lp.dist)> self.distance_limit or abs(lp.angle_rad) > self.angle_limit:
                return self.teacher
        return control_policy

    def _on_episode_done(self):
        self.alpha = self.p ** self._episode
        # Clear experience
        self._observations = []
        self._expert_actions = []

        InteractiveImitationLearning._on_episode_done(self)
