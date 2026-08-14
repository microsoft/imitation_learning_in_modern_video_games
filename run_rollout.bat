@echo off
setlocal enabledelayedexpansion

call .\.venv\Scripts\activate.bat

REM Play the trained agent in CS:GO. Usage:
REM   run_rollout.bat          -> RGB+D DINO (depth, ~24 FPS)
REM   run_rollout.bat rgb      -> RGB DINO   (no depth, ~155 FPS)
REM
REM PREREQS (or it hangs at "capturing mouse position"):
REM   1) legacy CS:GO running, spawned/alive in a match (Deathmatch -> Dust II)
REM   2) GSI connected -> check below shows ESTABLISHED (not just LISTENING).
REM      If only LISTENING: fully quit + relaunch CS:GO so it loads the GSI cfg.

set "MODE=%~1"
if "%MODE%"=="" set "MODE=rgbd"

echo.
echo === GSI check (port 3000) -- want ESTABLISHED, not just LISTENING ===
netstat -ano | findstr :3000
echo.

if /I "%MODE%"=="rgb" (
  echo Running RGB DINO rollout...
  python pixelbc\online_rollout\csgo_rgb_rollout.py --checkpoint training_logs\aim_expert_dino_vitb_seed_0\models\last.ckpt --fps 16 --ignore_keyboard_inputs %2 %3 %4
) else (
  echo Running RGB+D DINO rollout...
  python pixelbc\online_rollout\csgo_depth_rollout.py --checkpoint training_logs\aim_expert_dino_vitb_rgbd_seed_0\models\last.ckpt --fps 16 --depth_anything_dir C:\workspace\NAVIGATION\Depth-Anything-V2 --ignore_keyboard_inputs %2 %3 %4
)
