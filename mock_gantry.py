#!/usr/bin/env python3
"""Mock gantry client — responds to coordinator commands without hardware."""

import socket
import sys
import time

def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5555

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    print(f"Connected to coordinator at {host}:{port}")

    # Identify as gantry (same as move.py TCP mode)
    s.sendall(b"GANTRY\n")

    buf = ""
    try:
        while True:
            data = s.recv(4096)
            if not data:
                break
            buf += data.decode()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                cmd = line.strip()
                print(f"  CMD: {cmd}")

                if cmd == "PING":
                    s.sendall(b"PONG\n")

                elif cmd.startswith("MOVE "):
                    # Simulate instant move (no real motor travel)
                    s.sendall(b"MOTOR_STARTED\n")
                    time.sleep(3.0)  # brief fake delay
                    s.sendall(b"MOVE_COMPLETE\n")
                    print("    (mock move complete)")

                elif cmd.startswith("SPEED "):
                    s.sendall(b"SPEED_SET\n")

                elif cmd == "STOP":
                    s.sendall(b"STOPPED\n")

                elif cmd == "SHUTDOWN":
                    s.sendall(b"OK\n")
                    print("Shutdown received. Exiting.")
                    return

                else:
                    s.sendall(f"ERROR unknown: {cmd}\n".encode())

    except (KeyboardInterrupt, ConnectionError):
        print("Disconnected.")
    finally:
        s.close()

if __name__ == "__main__":
    main()