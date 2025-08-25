#!/usr/bin/env python3
"""
UDP command sending client for sending motion control commands to Isaac Lab simulation
"""

import socket
import sys
import time
import argparse


def send_command(host="localhost", port=12345, vx=0.0, vy=0.0, wz=0.0):
    """Send a single UDP command"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        message = f"{vx} {vy} {wz}"
        sock.sendto(message.encode('utf-8'), (host, port))
        sock.close()
        print(f"📤 Command sent: vx={vx}, vy={vy}, wz={wz}")
        return True
    except Exception as e:
        print(f"❌ Send failed: {e}")
        return False


def interactive_mode(host="localhost", port=12345):
    """Interactive mode, continuously accepts user input and sends commands"""
    print(f"\n=== Isaac Lab UDP Control Client ===")
    print(f"Target address: {host}:{port}")
    print("Input format: vx vy wz (space separated)")
    print("Where: vx=forward velocity, vy=lateral velocity, wz=angular velocity")
    print("Example: 1.0 0.0 0.2 (forward 1m/s, turn 0.2rad/s)")
    print("Enter 'q' or 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("Enter command (vx vy wz): ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
                
            # Parse input
            parts = user_input.split()
            if len(parts) != 3:
                print("⚠️  Format error, please enter three numbers: vx vy wz")
                continue
            
            vx, vy, wz = map(float, parts)
            
            # Send command
            if send_command(host, port, vx, vy, wz):
                print(f"✅ Command sent and applied")
            
        except ValueError:
            print("⚠️  Please enter valid numbers")
        except KeyboardInterrupt:
            print("\n👋 Program interrupted, goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def preset_commands(host="localhost", port=12345):
    """Preset command mode"""
    presets = {
        "1": (1.0, 0.0, 0.0, "Forward"),
        "2": (-1.0, 0.0, 0.0, "Backward"),
        "3": (0.0, 1.0, 0.0, "Left"),
        "4": (0.0, -1.0, 0.0, "Right"),
        "5": (0.0, 0.0, -0.5, "Turn Left"),
        "6": (0.0, 0.0, 0.5, "Turn Right"),
        "7": (1.0, 0.0, -0.2, "Forward+Turn Left"),
        "8": (1.0, 0.0, 0.2, "Forward+Turn Right"),
        "0": (0.0, 0.0, 0.0, "Stop"),
    }
    
    print(f"\n=== Isaac Lab UDP Control - Preset Command Mode ===")
    print(f"Target address: {host}:{port}")
    print("\nAvailable preset commands:")
    for key, (vx, vy, wz, desc) in presets.items():
        print(f"  {key}: {desc} (vx={vx}, vy={vy}, wz={wz})")
    print("  q: Exit\n")
    
    while True:
        try:
            choice = input("Select preset command (0-8, q to exit): ").strip()
            
            if choice.lower() in ['q', 'quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if choice in presets:
                vx, vy, wz, desc = presets[choice]
                if send_command(host, port, vx, vy, wz):
                    print(f"✅ Applied: {desc}")
            else:
                print("⚠️  Invalid selection, please enter 0-8 or q")
                
        except KeyboardInterrupt:
            print("\n👋 Program interrupted, goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Isaac Lab UDP Control Client")
    parser.add_argument("--host", default="localhost", help="Target host address")
    parser.add_argument("--port", type=int, default=12345, help="Target port")
    parser.add_argument("--mode", choices=["interactive", "preset", "single"], 
                       default="interactive", help="Running mode")
    parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity (single mode)")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity (single mode)")
    parser.add_argument("--wz", type=float, default=0.0, help="Angular velocity (single mode)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # Single send mode
        send_command(args.host, args.port, args.vx, args.vy, args.wz)
    elif args.mode == "preset":
        # Preset command mode
        preset_commands(args.host, args.port)
    else:
        # Interactive mode
        interactive_mode(args.host, args.port)


if __name__ == "__main__":
    main()
