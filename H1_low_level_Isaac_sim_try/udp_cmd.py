# udp_cmd.py (示意, 与原来一致) -----------------------------------------
import threading, socket, struct, numpy as np
class UDPCmdListener(threading.Thread):
    def __init__(self, env, port=5555):
        super().__init__(daemon=True)
        self.env = env
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", port))
        self.start()

    def run(self):
        while True:
            data, _ = self.sock.recvfrom(1024)
            cmd = np.array(struct.unpack("fff", data), np.float32)
            self.env.cmd = cmd
