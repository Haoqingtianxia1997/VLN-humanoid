#!/usr/bin/env python3
"""
UDP command sending client for wheeled robot control
Specialized for differential drive robots (Jetbot-style)
"""

import socket
import sys
import time
import argparse
import math


def send_command(host="localhost", port=12345, vx=0.0, wz=0.0):
    """Send a single UDP command for wheeled robot"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Format: "vx vy wz" where vy is always 0 for wheeled robots
        message = f"{vx} 0.0 {wz}"
        sock.sendto(message.encode('utf-8'), (host, port))
        sock.close()
        print(f"🚗 Wheeled robot command sent: vx={vx:.3f}m/s, wz={wz:.3f}rad/s")
        return True
    except Exception as e:
        print(f"❌ Send failed: {e}")
        return False


def interactive_mode(host="localhost", port=12345):
    """Interactive mode for wheeled robot control"""
    print(f"\n=== Wheeled Robot UDP Control Client ===")
    print(f"Target address: {host}:{port}")
    print("Input format: vx wz (space separated)")
    print("Where: vx=linear velocity (m/s), wz=angular velocity (rad/s)")
    print("Example: 0.5 0.2 (forward 0.5m/s, turn right 0.2rad/s)")
    print("Enter 'x' or 'quit' to exit")
    print("Controls:")
    print("  w: forward    s: backward")
    print("  a: turn left  d: turn right")
    print("  space: stop\n")
    
    while True:
        try:
            user_input = input("Enter command (vx wz) or key: ").strip().lower()
            
            if user_input in ['x', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            # Handle keyboard-style controls
            if user_input == 'w':
                vx, wz = 1.0, 0.0  # Forward
            elif user_input == 's':
                vx, wz = -1.0, 0.0  # Backward
            elif user_input == 'a':
                vx, wz = 0.0, 1.0  # Turn left
            elif user_input == 'd':
                vx, wz = 0.0, -1.0  # Turn right
            elif user_input == 'wa':
                vx, wz = 0.5, 0.3  # Forward + left
            elif user_input == 'wd':
                vx, wz = 0.5, -0.3  # Forward + right
            elif user_input == 'sa':
                vx, wz = -0.5, 0.3  # Backward + left
            elif user_input == 'sd':
                vx, wz = -0.5, -0.3  # Backward + right
            elif user_input in [' ', 'space', 'stop']:
                vx, wz = 0.0, 0.0  # Stop
            elif not user_input:
                continue
            else:
                # Parse numeric input
                parts = user_input.split()
                if len(parts) == 2:
                    vx, wz = map(float, parts)
                else:
                    print("⚠️  Format error, please enter two numbers: vx wz")
                    print("    Or use keys: w/s (forward/back), a/d (left/right), space (stop)")
                    continue
            
            # Send command
            if send_command(host, port, vx, wz):
                print(f"✅ Command applied: linear={vx:.3f}m/s, angular={wz:.3f}rad/s")
            
        except ValueError:
            print("⚠️  Please enter valid numbers or use control keys")
        except KeyboardInterrupt:
            print("\n👋 Program interrupted, goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def preset_commands(host="localhost", port=12345):
    """Preset command mode for wheeled robot"""
    presets = {
        "1": (0.5, 0.0, "Forward Slow"),
        "2": (1.0, 0.0, "Forward Fast"),
        "3": (-0.5, 0.0, "Backward Slow"),
        "4": (-1.0, 0.0, "Backward Fast"),
        "5": (0.0, 0.5, "Turn Left"),
        "6": (0.0, -0.5, "Turn Right"),
        "7": (0.0, 1.0, "Spin Left Fast"),
        "8": (0.0, -1.0, "Spin Right Fast"),
        "9": (0.5, 0.3, "Forward + Turn Left"),
        "a": (0.5, -0.3, "Forward + Turn Right"),
        "b": (-0.5, 0.3, "Backward + Turn Left"),
        "c": (-0.5, -0.3, "Backward + Turn Right"),
        "0": (0.0, 0.0, "Stop"),
    }
    
    print(f"\n=== Wheeled Robot UDP Control - Preset Command Mode ===")
    print(f"Target address: {host}:{port}")
    print("\nAvailable preset commands:")
    for key, (vx, wz, desc) in presets.items():
        print(f"  {key}: {desc} (vx={vx:.1f}m/s, wz={wz:.1f}rad/s)")
    print("  q: Exit\n")
    
    while True:
        try:
            choice = input("Select preset command: ").strip().lower()
            
            if choice in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if choice in presets:
                vx, wz, desc = presets[choice]
                if send_command(host, port, vx, wz):
                    print(f"✅ Applied: {desc}")
            else:
                print("⚠️  Invalid selection, please choose from the available options or 'q'")
                
        except KeyboardInterrupt:
            print("\n👋 Program interrupted, goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def trajectory_mode(host="localhost", port=12345):
    """Trajectory following mode for wheeled robot"""
    print(f"\n=== Wheeled Robot Trajectory Mode ===")
    print(f"Target address: {host}:{port}")
    print("Available trajectories:")
    print("  1: Circle (clockwise)")
    print("  2: Circle (counter-clockwise)")
    print("  3: Figure-8")
    print("  4: Square path")
    print("  5: Custom sine wave")
    print("  q: Exit\n")
    
    while True:
        try:
            choice = input("Select trajectory: ").strip().lower()
            
            if choice in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if choice == '1':
                # Circle clockwise
                print("🔄 Executing circular trajectory (clockwise)...")
                execute_circle_trajectory(host, port, clockwise=True)
            elif choice == '2':
                # Circle counter-clockwise
                print("🔄 Executing circular trajectory (counter-clockwise)...")
                execute_circle_trajectory(host, port, clockwise=False)
            elif choice == '3':
                # Figure-8
                print("∞ Executing figure-8 trajectory...")
                execute_figure8_trajectory(host, port)
            elif choice == '4':
                # Square path
                print("⬜ Executing square trajectory...")
                execute_square_trajectory(host, port)
            elif choice == '5':
                # Sine wave
                print("〰️ Executing sine wave trajectory...")
                execute_sine_trajectory(host, port)
            else:
                print("⚠️  Invalid selection")
                
        except KeyboardInterrupt:
            print("\n👋 Trajectory interrupted!")
            # Send stop command
            send_command(host, port, 0.0, 0.0)
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def execute_circle_trajectory(host, port, clockwise=True, radius=2.0, speed=0.5, duration=20.0):
    """Execute circular trajectory"""
    angular_velocity = speed / radius
    if not clockwise:
        angular_velocity = -angular_velocity
    
    print(f"Circle: radius={radius}m, speed={speed}m/s, duration={duration}s")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        send_command(host, port, speed, angular_velocity)
        time.sleep(0.1)
    
    # Stop
    send_command(host, port, 0.0, 0.0)
    print("✅ Circle trajectory completed")


def execute_figure8_trajectory(host, port, scale=2.0, duration=30.0):
    """Execute figure-8 trajectory"""
    print(f"Figure-8: scale={scale}, duration={duration}s")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        t = time.time() - start_time
        # Figure-8 parametric equations
        vx = 0.5
        wz = 0.8 * math.sin(2 * math.pi * t / 10.0)  # Period of 10 seconds
        
        send_command(host, port, vx, wz)
        time.sleep(0.1)
    
    # Stop
    send_command(host, port, 0.0, 0.0)
    print("✅ Figure-8 trajectory completed")


def execute_square_trajectory(host, port, side_length=3.0, speed=0.5):
    """Execute square trajectory"""
    print(f"Square: side={side_length}m, speed={speed}m/s")
    
    # Calculate time for each side
    side_time = side_length / speed
    turn_time = 1.5  # Time to turn 90 degrees
    
    for i in range(4):
        print(f"Side {i+1}/4")
        # Move forward
        start_time = time.time()
        while time.time() - start_time < side_time:
            send_command(host, port, speed, 0.0)
            time.sleep(0.1)
        
        # Turn 90 degrees (π/2 radians)
        print(f"Turning...")
        start_time = time.time()
        while time.time() - start_time < turn_time:
            send_command(host, port, 0.0, math.pi/2 / turn_time)
            time.sleep(0.1)
    
    # Stop
    send_command(host, port, 0.0, 0.0)
    print("✅ Square trajectory completed")


def execute_sine_trajectory(host, port, amplitude=1.0, frequency=0.2, duration=20.0):
    """Execute sine wave trajectory (forward with sinusoidal angular velocity)"""
    print(f"Sine wave: amplitude={amplitude}rad/s, frequency={frequency}Hz, duration={duration}s")
    
    start_time = time.time()
    while time.time() - start_time < duration:
        t = time.time() - start_time
        vx = 0.5  # Constant forward speed
        wz = amplitude * math.sin(2 * math.pi * frequency * t)
        
        send_command(host, port, vx, wz)
        time.sleep(0.1)
    
    # Stop
    send_command(host, port, 0.0, 0.0)
    print("✅ Sine wave trajectory completed")


def main():
    parser = argparse.ArgumentParser(description="Wheeled Robot UDP Control Client")
    parser.add_argument("--host", default="localhost", help="Target host address")
    parser.add_argument("--port", type=int, default=12345, help="Target port")
    parser.add_argument("--mode", choices=["interactive", "preset", "trajectory", "single"], 
                       default="interactive", help="Running mode")
    parser.add_argument("--vx", type=float, default=0.0, help="Linear velocity (single mode)")
    parser.add_argument("--wz", type=float, default=0.0, help="Angular velocity (single mode)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # Single send mode
        send_command(args.host, args.port, args.vx, args.wz)
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
