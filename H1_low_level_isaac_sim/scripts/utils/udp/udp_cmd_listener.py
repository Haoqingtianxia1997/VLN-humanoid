#!/usr/bin/env python3
"""
UDP command listener for real-time receiving robot motion control commands
"""

import socket
import threading
import numpy as np
import time
from typing import Optional, Tuple


class UDPCmdListener:
    """UDP command listener class, listens for UDP packets in background thread"""
    
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
        self._current_cmd = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._cmd_lock = threading.Lock()
        
        print(f"UDP listener initialized, listening address: {host}:{port}")
        print("Command format: 'vx vy wz' (space separated), example: '1.0 0.0 0.2'")
        print("Where vx=forward velocity, vy=lateral velocity, wz=angular velocity")
    
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
            
            print(f"✅ UDP listener started, listening on {self.host}:{self.port}")
            
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
        
        print("🛑 UDP listener stopped")
    
    def _listen_loop(self):
        """UDP listening loop (runs in background thread)"""
        print("🎧 Started listening for UDP commands...")
        
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
                print("    Correct format: 'vx vy wz', example: '1.0 0.0 0.2'")
                return
            
            vx, vy, wz = map(float, parts)
            
            # Build new command [vx, vy, heading=0.0, wz]
            new_cmd = np.array([vx, vy, 0.0, wz], dtype=np.float32)
            
            # Thread-safe update command
            with self._cmd_lock:
                self._current_cmd = new_cmd
            
            print(f"📡 Received new command (from {addr[0]}:{addr[1]}): vx={vx:.3f}, vy={vy:.3f}, wz={wz:.3f}")
            
        except ValueError as e:
            print(f"⚠️  Command parsing failed (from {addr[0]}:{addr[1]}): '{message}' - {e}")
            print("    Please ensure input is numeric, example: '1.0 0.0 0.2'")
    
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
    
    print("\n=== UDP Command Listener Test ===")
    print("Listener is running, you can send commands with:")
    print("1. Using netcat: echo '1.0 0.0 0.2' | nc -u localhost 12345")
    print("2. Using Python: send_udp_command(vx=1.0, vy=0.0, wz=0.2)")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            current = listener.get_current_cmd()
            print(f"\rCurrent command: vx={current[0]:.3f}, vy={current[1]:.3f}, wz={current[3]:.3f}", end="", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nExiting...")
        listener.stop()
