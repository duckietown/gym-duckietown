import numpy as np

from learning.imitation.iil_dagger.teacher import PurePursuitPolicy


def train_iil(environment, learner, horizon, episodes, alpha, max_velocity=0.7, _debug=False):
    # from IIL
    _horizon = horizon
    _episodes = episodes

    # data
    _observations = []
    _expert_actions = []

    teacher = PurePursuitPolicy(env=environment, ref_velocity=max_velocity, rescale_vel=[-1,max_velocity])

    # internal count
    _current_horizon = 0
    _episode = 0

    # event listeners
    _episode_done_listeners = []

    # expert decay
    p = alpha

    # thresholds used to give control back to learner once the teacher converges
    convergence_distance = 0.05
    convergence_angle = np.pi / 18

    # threshold on angle and distance from the lane when using the model to avoid going off track and env reset within an episode
    angle_limit = np.pi / 8
    distance_limit = 0.12

    def _query_expert(control_policy, control_action, observation):
        if control_policy == teacher:
            teacher_action = control_action
        else:
            teacher_action = teacher.predict(observation)

        if teacher_action is not None:
            #aggregate
            _observations.append(observation)
            _expert_actions.append(teacher_action)

        if teacher_action[0] < 0.1:
            _found_obstacle = True
        else:
            _found_obstacle = False

        return _found_obstacle

    def _mix(active_policy, _found_obstacle):
        control_policy = np.random.choice(
            a=[teacher, learner],
            p=[alpha, 1. - alpha]
        )
        if _found_obstacle:
            return teacher
        try:
            lp = environment.get_lane_pos2(environment.cur_pos, environment.cur_angle)
        except:
            return control_policy
        if active_policy:
            # keep using tecaher untill duckiebot converges back on track
            if not (abs(lp.dist) < convergence_distance and abs(lp.angle_rad) < convergence_angle):
                return teacher
        else:
            # in case we are using our learner and it started to diverge a lot we need to give
            # control back to the expert
            if abs(lp.dist) > distance_limit or abs(lp.angle_rad) > angle_limit:
                return teacher
        return control_policy

    # execute current control policy
    def _act(observation, active_policy, _found_obstacle):
        if _episode <= 1:  # initial policy equals expert's
            control_policy = teacher
        else:
            control_policy = _mix(active_policy, _found_obstacle)

        control_action = control_policy.predict(observation)

        _found_obstacle = _query_expert(control_policy, control_action, observation)

        return control_action, control_policy == teacher, _found_obstacle

    for _episode in range(_episodes):
        active_policy = True
        _found_obstacle = False
        done = False

        # sampling
        observation = environment.render_obs()
        for _current_horizon in range(_horizon):
            action, active_policy, _found_obstacle = _act(observation, active_policy, _found_obstacle)

            try:
                result = environment.step(action)
                next_observation, reward, done, info = result
            except Exception as e:
                print(e)
            if _debug:
                environment.render()
            if done:
                environment.reset()
            observation = next_observation

        # optimize
        learner.optimize(
            _observations, _expert_actions, _episode)
        print('saving model')
        learner.save()

        # Clear experience
        _observations = []
        _expert_actions = []
        alpha = p ** _episode
        environment.reset()

        if _episode % 5 == 0:
            learner.save(f"episode{_episode}.pt")
