import pystk
import numpy as np

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    target_velocity = 19.4175

    # Control acceleration
    if current_vel < target_velocity:
        action.acceleration = 1.0
    else:
        action.acceleration = 0.0

    # Control steering
    steer_angle = aim_point[0]
    action.steer = np.clip(steer_angle, -1, 1)

    # Control drifting
    if abs(steer_angle) > 0.20:
        action.drift = True
    else:
        action.drift = False

    distance_to_aim = np.linalg.norm(aim_point)
    if distance_to_aim > 0.40 and current_vel > 14.7:  # Brake if the aim point is far away and the kart is fast
        action.brake = True
    else:
        action.brake = False

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)

