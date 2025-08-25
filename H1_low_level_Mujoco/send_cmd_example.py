# send_cmd.py  —— 往 127.0.0.1:5555 发送 3×float32 (vx, vy, wz)

import socket, struct, time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
addr = ("127.0.0.1", 5555)

def send(vx, vy, wz):
    """把三个 float32 发送给仿真端"""
    sock.sendto(struct.pack("fff", vx, vy, wz), addr)

time.sleep(1)                           # 等仿真程序启动

# 1) 直走 3 秒
send(0.50, 0.00, 0.00)
time.sleep(3)

# 2) 原地左转 2 秒
send(0.00, 0.00, 0.60)
time.sleep(2)



# 4) 停止
send(0.00, 0.00, 0.00)
print("指令序列已发送完毕；可 Ctrl+C 退出。")
