#!/usr/bin/env python3
"""
UDP command listener for wheeled robot real-time control
Simplified version based on original udp_cmd_listener.py
"""

import socket
import threading
import numpy as np
import time
from typing import Optional, Tuple


class UDPCmdListener:
    """UDP command listener class for wheeled robot, compatible with original interface"""
    
    def __init__(self, host: str = "localhost", port: int = 12345):
        """
        Initialize UDP listener
        
        Args:
            host: Listening host address
            port: Listening port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.thread = None
        
        # Default command: [vx, vy, heading, wz] = [0.0, 0.0, 0.0, 0.0]
        # For wheeled robots, vy and heading are always 0
        self._current_cmd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._cmd_lock = threading.Lock()
        
        print(f"🚗 Wheeled Robot UDP listener initialized, listening address: {host}:{port}")
        print("Command format: 'vx vy wz' (space separated), vy is ignored")
        print("Where vx=forward velocity, wz=angular velocity")
        print("Example: '0.5 0.0 0.2' (forward 0.5m/s, turn 0.2rad/s)")
    
    def start(self):
        """Start UDP listening thread"""
        if self.running:
            print("UDP listener is already running")
            return
        
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(0.1)  # Set timeout for graceful exit
            
            self.running = True
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
            
            print(f"✅ Wheeled robot UDP listener started on {self.host}:{self.port}")
            
        except Exception as e:
            print(f"❌ UDP listener startup failed: {e}")
            self.running = False
    
    def stop(self):
        """Stop UDP listening"""
        if not self.running:
            return
        
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join(timeout=1.0)
        
        print("🛑 Wheeled robot UDP listener stopped")
    
    def _listen_loop(self):
        """UDP listening loop (runs in background thread)"""
        print("🎧 Started listening for wheeled robot UDP commands...")
        
        while self.running:
            try:
                # Receive UDP packet
                data, addr = self.socket.recvfrom(1024)
                message = data.decode('utf-8').strip()
                
                # Parse command
                self._parse_command(message, addr)
                
            except socket.timeout:
                # Timeout is normal, continue loop
                continue
            except Exception as e:
                if self.running:  # Only print error when normally running
                    print(f"⚠️  UDP receive error: {e}")
    
    def _parse_command(self, message: str, addr: Tuple[str, int]):
        """
        Parse received command string
        
        Args:
            message: Received message in format "vx vy wz"
            addr: Sender address
        """
        try:
            # Parse "vx vy wz" format
            parts = message.split()
            if len(parts) != 3:
                print(f"⚠️  Command format error (from {addr[0]}:{addr[1]}): '{message}'")
                print("    Correct format: 'vx vy wz', example: '0.5 0.0 0.2'")
                return
            
            vx, vy, wz = map(float, parts)
            
            # For wheeled robots, ignore vy and build command [vx, 0.0, 0.0, wz]
            new_cmd = np.array([vx, 0.0, 0.0, wz], dtype=np.float32)
            
            # Thread-safe update command
            with self._cmd_lock:
                self._current_cmd = new_cmd
            
            print(f"🚗 Wheeled robot command (from {addr[0]}:{addr[1]}): vx={vx:.3f}, wz={wz:.3f}")
            
        except ValueError as e:
            print(f"⚠️  Command parsing failed (from {addr[0]}:{addr[1]}): '{message}' - {e}")
            print("    Please ensure input is numeric, example: '0.5 0.0 0.2'")
    
    def get_current_cmd(self) -> np.ndarray:
        """
        Get current motion command
        
        Returns:
            Current command array [vx, vy, heading, wz]
        """
        with self._cmd_lock:
            return self._current_cmd.copy()


if __name__ == "__main__":
    # Test UDP listener
    import sys
    
    listener = UDPCmdListener()
    listener.start()
    
    print("\n=== Wheeled Robot UDP Command Listener Test ===")
    print("Listener is running, you can send commands with:")
    print("1. Using netcat: echo '0.5 0.0 0.2' | nc -u localhost 12345")
    print("2. Using wheeled client: python udp_cmd_client_wheeled.py")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            current = listener.get_current_cmd()
            print(f"\rCurrent command: vx={current[0]:.3f}, wz={current[3]:.3f}", end="", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nExiting...")
        listener.stop()
