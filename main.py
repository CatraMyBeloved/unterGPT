# scrape.py — anonymous Twitch chat logger, no auth needed
import socket, ssl, json, time

SERVER, PORT = "irc.chat.twitch.tv", 6697
NICK = "justinfan12345"      # anonymous read account, no token
CHANNEL = "#unter"    # lowercase, keep the leading #
OUT = "chat_logs.jsonl"

def parse_tags(s):
    out = {}
    for part in s.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out

def parse_line(line):
    # tagged PRIVMSG: "@tags :nick!user@host PRIVMSG #chan :message"
    tags = {}
    if line.startswith("@"):
        tag_str, line = line[1:].split(" ", 1)
        tags = parse_tags(tag_str)
    if "PRIVMSG" not in line:
        return None
    prefix, rest = line.split(" PRIVMSG ", 1)
    nick = prefix.split("!", 1)[0].lstrip(":")
    msg = rest.split(" :", 1)[1] if " :" in rest else ""
    # prefer server timestamp (ms) if present, else local time
    ts = int(tags.get("tmi-sent-ts", str(int(time.time()*1000)))) // 1000
    user = tags.get("display-name") or nick
    return {"ts": ts, "user": user, "msg": msg.strip()}

def connect():
    ctx = ssl.create_default_context()
    raw = socket.create_connection((SERVER, PORT))
    sock = ctx.wrap_socket(raw, server_hostname=SERVER)
    sock.sendall(b"CAP REQ :twitch.tv/tags\r\n")   # get timestamps + display names
    sock.sendall(f"NICK {NICK}\r\n".encode())
    sock.sendall(f"JOIN {CHANNEL}\r\n".encode())
    return sock

def run():
    sock = connect()
    buf = ""
    import os
    count = sum(1 for _ in open(OUT, encoding="utf-8")) if os.path.exists(OUT) else 0
    f = open(OUT, "a", encoding="utf-8")
    print(f"logging {CHANNEL} -> {OUT}")
    while True:
        try:
            buf += sock.recv(4096).decode("utf-8", errors="ignore")
        except Exception as e:
            print("recv error, reconnecting:", e)
            time.sleep(5); sock = connect(); buf = ""; continue
        lines = buf.split("\r\n")
        buf = lines.pop()
        for line in lines:
            if line.startswith("PING"):
                sock.sendall(b"PONG :tmi.twitch.tv\r\n")
                continue
            rec = parse_line(line)
            if rec and rec["msg"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                count += 1                        # NEW
                if count % 100 == 0:              # NEW
                    print(f"--- {count} messages saved ---")

if __name__ == "__main__":
    run()