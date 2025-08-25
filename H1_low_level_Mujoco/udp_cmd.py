# udp_cmd.py  —— 线程化 UDP 命令源
import socket, struct, threading, numpy as np

class UDPCmdListener:
    """
    在后台监听 3×float32 (vx, vy, wz)，写入 env.set_cmd()。
    默认端口 5555；发送端随便用 Python/Matlab/ROS/Unity 都行。
    """
    def __init__(self, env, port: int = 5555):
        self.env   = env
        self.port  = port
        self.sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)               # 非阻塞
        self._th   = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        print(f"[UDPCmd] 监听 0.0.0.0:{port} … 发送 12 字节 (3×float32) 即可控制")

    def _loop(self):
        import time, struct
        while True:
            try:
                dat, _ = self.sock.recvfrom(12)    # 恰好 3×float32 = 12 B
                vx, vy, wz = struct.unpack("fff", dat)
                self.env.set_cmd(np.array([vx, vy, wz], np.float32))
                # 打印到控制台便于调试
                print(f"[cmd] →  vx={vx:.3f}  vy={vy:.3f}  wz={wz:.3f}")
            except BlockingIOError:
                time.sleep(0.001)
