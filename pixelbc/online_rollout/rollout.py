import argparse


def start_minerl_rollout(args):
    from pixelbc.online_rollout.minerl_rollout import MineRLThreechopRollout

    rollout = MineRLThreechopRollout(
        checkpoint_path=args.checkpoint,
        save_dir=args.path,
        joystick_action_mode=args.joystick_action_mode,
        trigger_action_mode=args.trigger_action_mode,
        button_action_mode=args.button_action_mode,
        fps=args.fps,
        ignore_keyboard_inputs=args.ignore_keyboard_inputs,
    )

    rollout.rollout_loop()


def start_csgo_rollout(args):
    from pixelbc.online_rollout.csgo_rollout import CSGORollout

    rollout = CSGORollout(
        checkpoint_path=args.checkpoint,
        save_dir=args.path,
        joystick_action_mode=args.joystick_action_mode,
        trigger_action_mode=args.trigger_action_mode,
        button_action_mode=args.button_action_mode,
        fps=args.fps,
        ignore_keyboard_inputs=args.ignore_keyboard_inputs,
    )

    rollout.rollout_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout IL model in environment")
    parser.add_argument("-g", "--game", type=str, choices=["minerl", "csgo"], default="dungeons", help="Game to rollout model in")
    parser.add_argument("-ckpt", "--checkpoint", required=True, type=str, help="path to model checkpoint")
    parser.add_argument("-p", "--path", type=str, default="evaluation", help="Path to save evaluation results")
    parser.add_argument("-fps", "--fps", required=True, type=int, default=30, help="frames per second to provide to model")
    parser.add_argument(
        "-ja",
        "--joystick_action_mode",
        type=str,
        default="deterministic",
        choices=["stochastic", "deterministic"],
        help="Mode of action selection for joystick actions",
    )
    parser.add_argument(
        "-ta",
        "--trigger_action_mode",
        type=str,
        default="stochastic",
        choices=["stochastic", "deterministic"],
        help="Mode of action selection for trigger actions",
    )
    parser.add_argument(
        "-ba",
        "--button_action_mode",
        type=str,
        default="stochastic",
        choices=["stochastic", "deterministic"],
        help="Mode of action selection for button actions",
    )
    parser.add_argument("--ignore_keyboard_inputs", action="store_true", help="Ignore keyboard inputs and run for time limit")

    args = parser.parse_args()

    if args.game == "minerl":
        start_minerl_rollout(args)
    elif args.game == "csgo":
        start_csgo_rollout(args)
