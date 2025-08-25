#!/usr/bin/env python3
"""
UDP command sending client for H1 humanoid robot control
Supports full 3DOF control: vx (forward/backward), vy (left/right), wz (rotation)
"""

import socket
import sys
import time
import argparse
import math


def send_command(host="localhost", port=12345, vx=0.0, vy=0.0, wz=0.0):
    """Send a single UDP command for H1 humanoid robot"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Format: "vx vy wz" for full 3DOF control
        message = f"{vx} {vy} {wz}"
        sock.sendto(message.encode('utf-8'), (host, port))
        sock.close()
        print(f"🤖 H1 robot command sent: vx={vx:.3f}m/s, vy={vy:.3f}m/s, wz={wz:.3f}rad/s")
        return True
    except Exception as e:
        print(f"❌ Send failed: {e}")
        return False


def interactive_mode(host="localhost", port=12345):
    """Interactive mode for H1 humanoid robot control"""
    print(f"\n=== H1 Humanoid Robot UDP Control Client ===")
    print(f"Target address: {host}:{port}")
    print("Input format: vx vy wz (space separated)")
    print("Where: vx=forward/backward (m/s), vy=left/right (m/s), wz=rotation (rad/s)")
    print("Example: 0.5 0.2 0.1 (forward 0.5m/s, right 0.2m/s, turn right 0.1rad/s)")
    print("Enter 'x' or 'quit' to exit")
    print("\nKeyboard Controls:")
    print("  w: forward       s: backward")
    print("  a: left          d: right")
    print("  q: turn left     e: turn right")
    print("  space: stop")
    print("\nCombination controls:")
    print("  wa: forward+left    wd: forward+right")
    print("  sa: backward+left   sd: backward+right")
    print("  wq: forward+turn_left   we: forward+turn_right")
    print("  aq: left+turn_left      eq: right+turn_left")
    print("  ad: left+turn_right     ed: right+turn_right\n")
    
    while True:
        try:
            user_input = input("Enter command (vx vy wz) or key combination: ").strip().lower()
            
            if user_input in ['x', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            # Handle keyboard-style controls
            vx, vy, wz = 0.0, 0.0, 0.0  # Default values
            
            # Basic single key controls
            if user_input == 'w':
                vx = 1.0  # Forward
            elif user_input == 's':
                vx = -1.0  # Backward
            elif user_input == 'a':
                vy = 1.0  # Left
            elif user_input == 'd':
                vy = -1.0  # Right
            elif user_input == 'q':
                wz = 0.3  # Turn left
            elif user_input == 'e':
                wz = -0.3  # Turn right
            # Combination controls
            elif user_input == 'wa':
                vx, vy = 0.7, 0.7  # Forward + left
            elif user_input == 'wd':
                vx, vy = 0.7, -0.7  # Forward + right
            elif user_input == 'sa':
                vx, vy = -0.7, 0.7  # Backward + left
            elif user_input == 'sd':
                vx, vy = -0.7, -0.7  # Backward + right
            elif user_input == 'wq':
                vx, wz = 0.7, 0.3  # Forward + turn left
            elif user_input == 'we':
                vx, wz = 0.7, -0.3  # Forward + turn right
            elif user_input == 'sq':
                vx, wz = -0.7, 0.3  # Backward + turn left
            elif user_input == 'se':
                vx, wz = -0.7, -0.3  # Backward + turn right
            elif user_input == 'aq':
                vy, wz = 0.7, 0.3  # Left + turn left
            elif user_input == 'eq':
                vy, wz = -0.7, 0.3  # Right + turn left
            elif user_input == 'ad':
                vy, wz = 0.7, -0.5  # Left + turn right
            elif user_input == 'ed':
                vy, wz = -0.7, -0.5  # Right + turn right
            elif user_input in [' ', 'space', 'stop']:
                vx, vy, wz = 0.0, 0.0, 0.0  # Stop
            elif not user_input:
                continue
            else:
                # Parse numeric input
                parts = user_input.split()
                if len(parts) == 3:
                    vx, vy, wz = map(float, parts)
                else:
                    print("⚠️  Format error, please enter three numbers: vx vy wz")
                    print("    Or use keys: w/s (forward/back), a/d (left/right), q/e (turn), space (stop)")
                    continue
            
            # Send command
            if send_command(host, port, vx, vy, wz):
                print(f"✅ Command applied: forward={vx:.3f}m/s, lateral={vy:.3f}m/s, angular={wz:.3f}rad/s")
            
        except ValueError:
            print("⚠️  Please enter valid numbers or use control keys")
        except KeyboardInterrupt:
            print("\n👋 Program interrupted, goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def preset_commands(host="localhost", port=12345):
    """Preset command mode for H1 humanoid robot"""
    presets = {
        "1": (0.5, 0.0, 0.0, "Forward Slow"),
        "2": (1.0, 0.0, 0.0, "Forward Fast"),
        "3": (-0.5, 0.0, 0.0, "Backward Slow"),
        "4": (-1.0, 0.0, 0.0, "Backward Fast"),
        "5": (0.0, 0.5, 0.0, "Left Slow"),
        "6": (0.0, 1.0, 0.0, "Left Fast"),
        "7": (0.0, -0.5, 0.0, "Right Slow"),
        "8": (0.0, -1.0, 0.0, "Right Fast"),
        "9": (0.0, 0.0, 0.5, "Turn Left"),
        "a": (0.0, 0.0, -0.5, "Turn Right"),
        "b": (0.5, 0.5, 0.0, "Forward + Left"),
        "c": (0.5, -0.5, 0.0, "Forward + Right"),
        "d": (-0.5, 0.5, 0.0, "Backward + Left"),
        "e": (-0.5, -0.5, 0.0, "Backward + Right"),
        "f": (0.5, 0.0, 0.3, "Forward + Turn Left"),
        "g": (0.5, 0.0, -0.3, "Forward + Turn Right"),
        "h": (0.0, 0.5, 0.3, "Left + Turn Left"),
        "i": (0.0, -0.5, -0.3, "Right + Turn Right"),
        "0": (0.0, 0.0, 0.0, "Stop"),
    }
    
    print(f"\n=== H1 Humanoid Robot UDP Control - Preset Command Mode ===")
    print(f"Target address: {host}:{port}")
    print("\nAvailable preset commands:")
    for key, (vx, vy, wz, desc) in presets.items():
        print(f"  {key}: {desc} (vx={vx:.1f}m/s, vy={vy:.1f}m/s, wz={wz:.1f}rad/s)")
    print("  q: Exit\n")
    
    while True:
        try:
            choice = input("Select preset command: ").strip().lower()
            
            if choice in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if choice in presets:
                vx, vy, wz, desc = presets[choice]
                if send_command(host, port, vx, vy, wz):
                    print(f"✅ Applied: {desc}")
            else:
                print("⚠️  Invalid selection, please choose from the available options or 'q'")
                
        except KeyboardInterrupt:
            print("\n👋 Program interrupted, goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def trajectory_mode(host="localhost", port=12345):
    """Trajectory following mode for H1 humanoid robot"""
    print(f"\n=== H1 Humanoid Robot Trajectory Mode ===")
    print(f"Target address: {host}:{port}")
    print("Available trajectories:")
    print("  1: Circle walking (clockwise)")
    print("  2: Circle walking (counter-clockwise)")
    print("  3: Figure-8 walking")
    print("  4: Square path")
    print("  5: Lateral oscillation (side-to-side)")
    print("  6: Complex omnidirectional pattern")
    print("  q: Exit\n")
    
    while True:
        try:
            choice = input("Select trajectory: ").strip().lower()
            
            if choice in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if choice == '1':
                # Circle clockwise
                print("🔄 Executing circular walking trajectory (clockwise)...")
                execute_circle_trajectory(host, port, clockwise=True)
            elif choice == '2':
                # Circle counter-clockwise
                print("🔄 Executing circular walking trajectory (counter-clockwise)...")
                execute_circle_trajectory(host, port, clockwise=False)
            elif choice == '3':
                # Figure-8
                print("∞ Executing figure-8 walking trajectory...")
                execute_figure8_trajectory(host, port)
            elif choice == '4':
                # Square path
                print("⬜ Executing square walking trajectory...")
                execute_square_trajectory(host, port)
            elif choice == '5':
                # Lateral oscillation
                print("↔️ Executing lateral oscillation...")
                execute_lateral_oscillation(host, port)
            elif choice == '6':
                # Complex omnidirectional
                print("🌀 Executing complex omnidirectional pattern...")
                execute_complex_pattern(host, port)
            else:
                print("⚠️  Invalid selection")
                
        except KeyboardInterrupt:
            print("\n👋 Trajectory interrupted!")
            # Send stop command
            send_command(host, port, 0.0, 0.0, 0.0)
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def execute_circle_trajectory(host, port, clockwise=True, radius=2.0, speed=0.3, duration=20.0):
    """Execute circular walking trajectory"""
    angular_velocity = speed / radius
    if not clockwise:
        angular_velocity = -angular_velocity
    
    print(f"Circle walking: radius={radius}m, speed={speed}m/s, duration={duration}s")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        send_command(host, port, speed, 0.0, angular_velocity)
        time.sleep(0.1)
    
    # Stop
    send_command(host, port, 0.0, 0.0, 0.0)
    print("✅ Circle walking trajectory completed")


def execute_figure8_trajectory(host, port, scale=1.5, duration=30.0):
    """Execute figure-8 walking trajectory"""
    print(f"Figure-8 walking: scale={scale}, duration={duration}s")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        t = time.time() - start_time
        # Figure-8 parametric equations for humanoid
        vx = 0.3
        vy = 0.2 * math.sin(4 * math.pi * t / 15.0)  # Lateral component
        wz = 0.6 * math.sin(2 * math.pi * t / 15.0)  # Rotation component
        
        send_command(host, port, vx, vy, wz)
        time.sleep(0.1)
    
    # Stop
    send_command(host, port, 0.0, 0.0, 0.0)
    print("✅ Figure-8 walking trajectory completed")


def execute_square_trajectory(host, port, side_length=2.0, speed=0.3):
    """Execute square walking trajectory"""
    print(f"Square walking: side={side_length}m, speed={speed}m/s")
    
    # Calculate time for each side
    side_time = side_length / speed
    turn_time = 2.0  # Time to turn 90 degrees (slower for humanoid)
    
    for i in range(4):
        print(f"Side {i+1}/4")
        # Move forward
        start_time = time.time()
        while time.time() - start_time < side_time:
            send_command(host, port, speed, 0.0, 0.0)
            time.sleep(0.1)
        
        # Turn 90 degrees (π/2 radians)
        print(f"Turning...")
        start_time = time.time()
        while time.time() - start_time < turn_time:
            send_command(host, port, 0.0, 0.0, math.pi/2 / turn_time)
            time.sleep(0.1)
    
    # Stop
    send_command(host, port, 0.0, 0.0, 0.0)
    print("✅ Square walking trajectory completed")


def execute_lateral_oscillation(host, port, amplitude=0.5, frequency=0.3, duration=15.0):
    """Execute lateral oscillation (side-to-side walking)"""
    print(f"Lateral oscillation: amplitude={amplitude}m/s, frequency={frequency}Hz, duration={duration}s")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        t = time.time() - start_time
        vx = 0.2  # Slow forward movement
        vy = amplitude * math.sin(2 * math.pi * frequency * t)  # Side-to-side
        wz = 0.0
        
        send_command(host, port, vx, vy, wz)
        time.sleep(0.1)
    
    # Stop
    send_command(host, port, 0.0, 0.0, 0.0)
    print("✅ Lateral oscillation completed")


def execute_complex_pattern(host, port, duration=25.0):
    """Execute complex omnidirectional walking pattern"""
    print(f"Complex omnidirectional pattern: duration={duration}s")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        t = time.time() - start_time
        # Complex pattern using multiple sinusoidal components
        vx = 0.3 * math.sin(2 * math.pi * t / 8.0)
        vy = 0.2 * math.cos(2 * math.pi * t / 6.0)
        wz = 0.4 * math.sin(2 * math.pi * t / 10.0)
        
        send_command(host, port, vx, vy, wz)
        time.sleep(0.1)
    
    # Stop
    send_command(host, port, 0.0, 0.0, 0.0)
    print("✅ Complex omnidirectional pattern completed")


def main():
    parser = argparse.ArgumentParser(description="H1 Humanoid Robot UDP Control Client")
    parser.add_argument("--host", default="localhost", help="Target host address")
    parser.add_argument("--port", type=int, default=12345, help="Target port")
    parser.add_argument("--mode", choices=["interactive", "preset", "trajectory", "single"], 
                       default="interactive", help="Running mode")
    parser.add_argument("--vx", type=float, default=0.0, help="Forward/backward velocity (single mode)")
    parser.add_argument("--vy", type=float, default=0.0, help="Left/right velocity (single mode)")
    parser.add_argument("--wz", type=float, default=0.0, help="Angular velocity (single mode)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # Single send mode
        send_command(args.host, args.port, args.vx, args.vy, args.wz)
    elif args.mode == "preset":
        # Preset command mode
        preset_commands(args.host, args.port)
    elif args.mode == "trajectory":
        # Trajectory mode
        trajectory_mode(args.host, args.port)
    else:
        # Interactive mode
        interactive_mode(args.host, args.port)


if __name__ == "__main__":
    main()
