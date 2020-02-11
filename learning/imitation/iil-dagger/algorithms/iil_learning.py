

class InteractiveImitationLearning:
    """
    A class used to contain main imitation learning algorithm
    ...
    Methods
    -------
    train(samples, debug)
        start training imitation learning
    """
    def __init__(self, env, teacher, learner, horizon, episodes, test=False):
        """
        Parameters
        ----------
        env : 
            duckietown environment
        teacher : 
            expert used to train imitation learning
        learner : 
            model used to learn
        horizon : int
            which is the number of observations to be collected during one episode
        episode : int
            number of episodes which is the number of collected trajectories
        """

        self.environment = env
        self.teacher = teacher
        self.learner = learner
        self.test = test

        # from IIL
        self._horizon = horizon
        self._episodes = episodes

        # data
        self._observations = []
        self._expert_actions = []

        # statistics
        self.learner_action = None
        self.learner_uncertainty = None

        self.teacher_action = None
        self.active_policy = True  # if teacher is active

        # internal count
        self._current_horizon = 0
        self._episode = 0

        # event listeners
        self._episode_done_listeners = []
        self._found_obstacle = False
        # steering angle gain
        self.gain = 10

    def train(self, debug=False):
        """
        Parameters
        ----------
        teacher : 
            expert used to train imitation learning
        learner : 
            model used to learn
        horizon : int
            which is the number of observations to be collected during one episode
        episode : int
            number of episodes which is the number of collected trajectories
        """
        self._debug = debug
        for episode in range(self._episodes):
            self._episode = episode
            self._sampling()
            self._optimize()  # episodic learning
            self._on_episode_done()

    def _sampling(self):
        observation = self.environment.render_obs()
        for horizon in range(self._horizon):
            self._current_horizon = horizon
            action = self._act(observation)
            try:
                next_observation, reward, done, info = self.environment.step([action[0], action[1]*self.gain])
            except Exception as e:
                print(e)
            if self._debug:
                self.environment.render()
            observation = next_observation

    # execute current control policy
    def _act(self, observation):
        if self._episode <= 1:  # initial policy equals expert's
            control_policy = self.teacher
        else:
            control_policy = self._mix()

        control_action = control_policy.predict(observation)

        self._query_expert(control_policy, control_action,observation)

        self.active_policy = control_policy == self.teacher
        if self.test:
            return self.learner_action

        return control_action

    def _query_expert(self, control_policy, control_action, observation):
        if control_policy == self.learner:
            self.learner_action = control_action
        else:
            self.learner_action = self.learner.predict(observation)

        if control_policy == self.teacher:
            self.teacher_action = control_action
        else:
            self.teacher_action = self.teacher.predict(observation)

        if self.teacher_action is not None:
            self._aggregate(observation, self.teacher_action)

        if self.teacher_action[0] < 0.1:
            self._found_obstacle = True
        else:
            self._found_obstacle = False

    def _mix(self):
        raise NotImplementedError()

    def _aggregate(self, observation, action):
        if not(self.test):
            self._observations.append(observation)
            self._expert_actions.append(action)

    def _optimize(self):
        if not(self.test):
            self.learner.optimize(
                self._observations, self._expert_actions, self._episode)
            print('saving model')
            self.learner.save()

    # TRAINING EVENTS

    # triggered after an episode of learning is done
    def on_episode_done(self, listener):
        self._episode_done_listeners.append(listener)

    def _on_episode_done(self):
        for listener in self._episode_done_listeners:
            listener.episode_done(self._episode)
        self.environment.reset()
