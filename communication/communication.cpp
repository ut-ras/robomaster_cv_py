// communication.cpp
// Build: g++ -std=c++17 communication.cpp -o comm

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// -------------------- CRC helpers (reflect-aware) --------------------

static inline uint8_t reflect8(uint8_t v) {
    v = (v & 0xF0) >> 4 | (v & 0x0F) << 4;
    v = (v & 0xCC) >> 2 | (v & 0x33) << 2;
    v = (v & 0xAA) >> 1 | (v & 0x55) << 1;
    return v;
}

static inline uint16_t reflect16(uint16_t v) {
    v = (v & 0xFF00) >> 8 | (v & 0x00FF) << 8;
    v = (v & 0xF0F0) >> 4 | (v & 0x0F0F) << 4;
    v = (v & 0xCCCC) >> 2 | (v & 0x3333) << 2;
    v = (v & 0xAAAA) >> 1 | (v & 0x5555) << 1;
    return v;
}

// Python config (crc8):
// width=8, poly=0x31, init=0xFF, xorout=0x00, refin=True, refout=True
uint8_t crc8_calc(const uint8_t* data, size_t len) {
    const uint8_t poly = 0x31;
    uint8_t crc = 0xFF;
    for (size_t i = 0; i < len; ++i) {
        uint8_t byte = reflect8(data[i]);     // refin=True
        crc ^= byte;
        for (int b = 0; b < 8; ++b) {
            if (crc & 0x80) crc = (crc << 1) ^ poly;
            else            crc <<= 1;
        }
    }
    crc = reflect8(crc);                      // refout=True
    crc ^= 0x00;                              // xorout
    return crc;
}

// Python config (crc16):
// width=16, poly=0x1021, init=0xFFFF, xorout=0x0000, refin=True, refout=True
uint16_t crc16_calc(const uint8_t* data, size_t len) {
    const uint16_t poly = 0x1021;
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        uint8_t byte = reflect8(data[i]);     // refin=True
        crc ^= static_cast<uint16_t>(byte) << 8;
        for (int b = 0; b < 8; ++b) {
            if (crc & 0x8000) crc = (crc << 1) ^ poly;
            else              crc <<= 1;
        }
    }
    crc = reflect16(crc);                     // refout=True
    crc ^= 0x0000;                            // xorout
    return crc;
}

// -------------------- Message types --------------------
enum class MessageType : uint16_t {
    CMD_Odometry_Data = 1,
    CMD_Turret_Aim    = 2,
};

// -------------------- Little-endian packers --------------------
static inline void put_u8 (std::vector<uint8_t>& v, uint8_t x)  { v.push_back(x); }
static inline void put_u16(std::vector<uint8_t>& v, uint16_t x) { v.push_back(uint8_t(x)); v.push_back(uint8_t(x >> 8)); }
static inline void put_f32(std::vector<uint8_t>& v, float f)    {
    static_assert(sizeof(float) == 4, "float must be 32-bit");
    uint32_t u; std::memcpy(&u, &f, 4);
    v.push_back(uint8_t(u)); v.push_back(uint8_t(u >> 8));
    v.push_back(uint8_t(u >> 16)); v.push_back(uint8_t(u >> 24));
}
static inline void put_bool(std::vector<uint8_t>& v, bool b)    { v.push_back(b ? 1 : 0); }

// -------------------- UART wrapper --------------------
class SerialPort {
    int fd_ = -1;
public:
    ~SerialPort() { if (fd_ >= 0) ::close(fd_); }

    void openPort(const char* path = "/dev/ttyS4", int baud = B115200, double timeout_sec = 0.5) {
        fd_ = ::open(path, O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) throw std::runtime_error(std::string("open failed: ") + path);

        termios tio{}; tcgetattr(fd_, &tio);
        cfmakeraw(&tio);
        // 8N1
        tio.c_cflag &= ~PARENB;
        tio.c_cflag &= ~CSTOPB;
        tio.c_cflag &= ~CSIZE;
        tio.c_cflag |= CS8;
        // baud
        cfsetispeed(&tio, baud);
        cfsetospeed(&tio, baud);
        // read timeout: VTIME in deciseconds
        tio.c_cc[VTIME] = static_cast<cc_t>(std::lround(timeout_sec * 10.0)); // 0.5s -> 5
        tio.c_cc[VMIN]  = 0; // return as soon as available or timeout
        tcsetattr(fd_, TCSANOW, &tio);

        // make blocking after setting timeout semantics
        int flags = fcntl(fd_, F_GETFL, 0);
        fcntl(fd_, F_SETFL, flags & ~O_NONBLOCK);
    }

    ssize_t writeAll(const void* buf, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(buf);
        size_t sent = 0;
        while (sent < n) {
            ssize_t w = ::write(fd_, p + sent, n - sent);
            if (w < 0) return w;
            sent += size_t(w);
        }
        return ssize_t(sent);
    }

    // Simple "read a line" like pyserial.readline() (ends with '\n' or timeout)
    std::string readLine() {
        std::string out;
        uint8_t ch;
        while (true) {
            ssize_t r = ::read(fd_, &ch, 1);
            if (r == 1) {
                out.push_back(char(ch));
                if (ch == '\n') break;
            } else {
                // timeout (VTIME) or no data
                break;
            }
        }
        return out;
    }
};

// -------------------- Protocol packing --------------------
// Python layout:
// FrameHeader: <B H B   (head=0xA5, data_len, seq)
// message_no_crc16: < [header bytes]  B  H  [data bytes]
//                    (crc8(header))  (u16 message_type)
// Full message: message_no_crc16 + <H crc16(message_no_crc16)>
struct TurretData {
    float xPos, yPos, zPos;
    float xVel, yVel, zVel;
    float xAcc, yAcc, zAcc;
    bool  hasTarget;
};

class Communication {
    SerialPort ser_;
    uint8_t seq_ = 0;

    static std::vector<uint8_t> packFrame(const std::vector<uint8_t>& data, MessageType type, uint8_t seq) {
        const uint8_t FRAME_HEAD = 0xA5;
        uint16_t data_len = static_cast<uint16_t>(data.size());

        // FrameHeader <BHB>
        std::vector<uint8_t> header;
        header.reserve(4);
        put_u8(header, FRAME_HEAD);
        put_u16(header, data_len);
        put_u8(header, seq);

        // CRC8 over header
        uint8_t crc8 = crc8_calc(header.data(), header.size());

        // message_no_crc16: <header bytes> + <B crc8> + <H type> + <data bytes>
        std::vector<uint8_t> msg_no_crc;
        msg_no_crc.reserve(header.size() + 1 + 2 + data.size());
        msg_no_crc.insert(msg_no_crc.end(), header.begin(), header.end());
        put_u8(msg_no_crc, crc8);
        put_u16(msg_no_crc, static_cast<uint16_t>(type));
        msg_no_crc.insert(msg_no_crc.end(), data.begin(), data.end());

        // CRC16 over message_no_crc16
        uint16_t c16 = crc16_calc(msg_no_crc.data(), msg_no_crc.size());

        // full_message = message_no_crc16 + <H crc16>
        std::vector<uint8_t> full = msg_no_crc;
        put_u16(full, c16);
        return full;
    }

    static std::vector<uint8_t> packTurretData(const TurretData& t) {
        // Matches struct.Struct("<fffffffff?") in Python
        std::vector<uint8_t> v;
        v.reserve(9*4 + 1);
        put_f32(v, t.xPos); put_f32(v, t.yPos); put_f32(v, t.zPos);
        put_f32(v, t.xVel); put_f32(v, t.yVel); put_f32(v, t.zVel);
        put_f32(v, t.xAcc); put_f32(v, t.yAcc); put_f32(v, t.zAcc);
        put_bool(v, t.hasTarget);
        return v;
    }

public:
    void initialize_communication() {
        ser_.openPort("/dev/ttyS4", B115200, 0.5);
    }

    std::string read_message() {
        return ser_.readLine(); // textual lines (if peer sends any)
    }

    void send_turret_data(float xPos, float yPos, float zPos,
                          float xVel, float yVel, float zVel,
                          float xAcc, float yAcc, float zAcc,
                          bool hasTarget)
    {
        TurretData t { xPos,yPos,zPos, xVel,yVel,zVel, xAcc,yAcc,zAcc, hasTarget };
        auto payload = packTurretData(t);
        auto frame   = packFrame(payload, MessageType::CMD_Turret_Aim, seq_++);
        if (ser_.writeAll(frame.data(), frame.size()) < 0) {
            throw std::runtime_error("write failed");
        }
    }
};

// -------------------- Example usage --------------------
int main() {
    try {
        Communication com;
        com.initialize_communication();

        // Example: send zeros with hasTarget=false
        com.send_turret_data(0,0,0, 0,0,0, 0,0,0, false);

        // Example: read a line (if the other side prints lines)
        std::string msg = com.read_message();
        if (!msg.empty()) std::cerr << "RX: " << msg << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

