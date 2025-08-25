#include "unitree_h1_hardware_interface/motor_crc.h"


void get_crc(unitree_hg::msg::LowCmd& msg) {
    // 1. Create a buffer to hold the serialized message.
    std::vector<uint8_t> buffer;
    buffer.reserve(512); // Estimate a size for the buffer, adjust as needed.

    auto appendToBuffer = [&](const void* data, size_t size) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), bytes, bytes + size);
    };

    // 2. Serialize the message fields into the buffer.
    appendToBuffer(&msg.mode_pr, sizeof(msg.mode_pr));
    appendToBuffer(&msg.mode_machine, sizeof(msg.mode_machine));

    for (const auto& cmd : msg.motor_cmd) {
        appendToBuffer(&cmd.mode, sizeof(cmd.mode));
        appendToBuffer(&cmd.q, sizeof(cmd.q));
        appendToBuffer(&cmd.dq, sizeof(cmd.dq));
        appendToBuffer(&cmd.tau, sizeof(cmd.tau));
        appendToBuffer(&cmd.kp, sizeof(cmd.kp));
        appendToBuffer(&cmd.kd, sizeof(cmd.kd));
        appendToBuffer(&cmd.reserve, sizeof(cmd.reserve));
    }

    appendToBuffer(msg.reserve.data(), msg.reserve.size() * sizeof(uint32_t));

    // 3. Führe die CRC-Berechnung auf dem Buffer aus.
    uint32_t crc = crc32_core(reinterpret_cast<uint32_t*>(buffer.data()), buffer.size() / 4);

    // 4. Write the CRC back to the message.
    msg.crc = crc;
}


uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
    uint32_t xbit = 0;
    uint32_t data = 0;
    uint32_t CRC32 = 0xFFFFFFFF;
    const uint32_t dwPolynomial = 0x04c11db7;
    
    for (uint32_t i = 0; i < len; i++) {
        xbit = 1 << 31;
        data = ptr[i];

        for (uint32_t bits = 0; bits < 32; bits++) {
            if (CRC32 & 0x80000000) {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
                CRC32 <<= 1;
            if (data & xbit)
                CRC32 ^= dwPolynomial;

            xbit >>= 1;
        }
    }
    return CRC32;
}