import socket

class UdpEmitter:
    def __init__(self, target_ip: str = "127.0.0.1", target_port: int = 7777):
        self.target_ip = target_ip
        self.target_port = target_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    def send(self, data: bytes):
        try:
            self.sock.sendto(data, (self.target_ip, self.target_port))
        except Exception as e:
            print(f"UDP Send Error: {e}")

    def close(self):
        self.sock.close()
