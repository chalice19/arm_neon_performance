#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <cstring>
#include "benchmark/benchmark.h"

bool MoveMaskSSE2NEON(uint8x16_t cmp_res) {
  uint16x8_t high_bits = vreinterpretq_u16_u8(vshrq_n_u8(cmp_res, 7));
  uint32x4_t paired16 =
      vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 7));
  uint64x2_t paired32 =
      vreinterpretq_u64_u32(vsraq_n_u32(paired16, paired16, 14));
  uint8x16_t paired64 =
      vreinterpretq_u8_u64(vsraq_n_u64(paired32, paired32, 28));
  uint32_t mask =
      vgetq_lane_u8(paired64, 0) | ((int)vgetq_lane_u8(paired64, 8) << 8);
  return !mask;
}
static void BM_IsAsciiSSE2Neon(benchmark::State& state) {
  std::unique_ptr<unsigned char[]> buf(new unsigned char[1024 * 1024]);
  for (size_t i = 0; i < 1024*1024; ++i) {
    buf[i] = 'a';
  }

  for (auto _ : state) {
    bool res = false;
    for (size_t i = 0; i < 1024*1024; i += 32) {
      res &= MoveMaskSSE2NEON(vld1q_u8(buf.get() + i));
      res &= MoveMaskSSE2NEON(vld1q_u8(buf.get() + i + 16));
      benchmark::DoNotOptimize(buf.get());
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_IsAsciiSSE2Neon);

bool only_ascii_in_vector(uint8x16_t input)
{
    uint8x16_t and_result = vandq_u8(input, vdupq_n_u8(0x80));
    uint8x8_t or_result = vorr_u8(vget_low_u8(and_result), vget_high_u8(and_result));
    return !vget_lane_u64(vreinterpret_u64_u8(or_result), 0);
}
static void BM_IsAscii(benchmark::State& state) {
  std::unique_ptr<unsigned char[]> buf(new unsigned char[1024 * 1024]);
  for (size_t i = 0; i < 1024*1024; ++i) {
    buf[i] = 'a';
  }

  for (auto _ : state) {
    bool res = false;
    for (size_t i = 0; i < 1024*1024; i += 32) {
      res &= only_ascii_in_vector(vld1q_u8(buf.get() + i));
      res &= only_ascii_in_vector(vld1q_u8(buf.get() + i + 16));
      benchmark::DoNotOptimize(buf.get());
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_IsAscii);

bool only_ascii_in_vector_64(uint64x2_t input)
{
    uint64x2_t and_result = vandq_u64(input, vdupq_n_u64(0x8080808080808080));
    uint64x1_t or_result = vorr_u64(vget_low_u64(and_result), vget_high_u64(and_result));
    return !vget_lane_u64(or_result, 0);
}
static void BM_IsAscii_64(benchmark::State& state) {
  std::unique_ptr<unsigned char[]> buf(new unsigned char[1024 * 1024]);
  for (size_t i = 0; i < 1024*1024; ++i) {
    buf[i] = 'a';
  }

  for (auto _ : state) {
    bool res = false;
    for (size_t i = 0; i < 1024*1024; i += 32) {
      res &= only_ascii_in_vector_64(vld1q_u64((const uint64_t*)(buf.get() + i)));
      res &= only_ascii_in_vector_64(vld1q_u64((const uint64_t*)(buf.get() + i + 16)));
      benchmark::DoNotOptimize(buf.get());
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_IsAscii_64);

bool only_ascii_in_vector_max(uint8x16_t input)
{
    return !vmaxvq_u8(vandq_u8(input, vdupq_n_u8(0x80)));
}
static void BM_IsAscii_max(benchmark::State& state) {
  std::unique_ptr<unsigned char[]> buf(new unsigned char[1024 * 1024]);
  for (size_t i = 0; i < 1024*1024; ++i) {
    buf[i] = 'a';
  }

  for (auto _ : state) {
    bool res = false;
    for (size_t i = 0; i < 1024*1024; i += 32) {
      res &= only_ascii_in_vector_max(vld1q_u8(buf.get() + i));
      res &= only_ascii_in_vector_max(vld1q_u8(buf.get() + i + 16));
      benchmark::DoNotOptimize(buf.get());
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_IsAscii_max);

bool only_ascii_in_vector_max_32(uint32x4_t input)
{
    return !vmaxvq_u32(vandq_u32(input, vdupq_n_u32(0x80808080)));
}
static void BM_IsAscii_max_32(benchmark::State& state) {
  std::unique_ptr<unsigned char[]> buf(new unsigned char[1024 * 1024]);
  for (size_t i = 0; i < 1024*1024; ++i) {
    buf[i] = 'a';
  }

  for (auto _ : state) {
    bool res = false;
    for (size_t i = 0; i < 1024*1024; i += 32) {
      res &= only_ascii_in_vector_max_32(vld1q_u32((const uint32_t*)(buf.get() + i)));
      res &= only_ascii_in_vector_max_32(vld1q_u32((const uint32_t*)(buf.get() + i + 16)));
      benchmark::DoNotOptimize(buf.get());
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_IsAscii_max_32);

bool only_ascii_in_vector_ccmp(uint64x2_t input)
{
    uint64x2_t and_result = vandq_u64(input, vdupq_n_u64(0x8080808080808080));
    uint64x1_t lo = vget_low_u64(and_result);
    uint64x1_t hi = vget_high_u64(and_result);
    return (uint64_t)lo != 0 ? false : (uint64_t)hi == 0;
}

static void BM_IsAsciiCCmp(benchmark::State& state) {
  std::unique_ptr<unsigned char[]> buf(new unsigned char[1024 * 1024]);
  for (size_t i = 0; i < 1024*1024; ++i) {
    buf[i] = 'a';
  }

  for (auto _ : state) {
    bool res = false;
    for (size_t i = 0; i < 1024*1024; i += 32) {
      res &= only_ascii_in_vector_ccmp(vld1q_u64((const uint64_t*)(buf.get() + i)));
      res &= only_ascii_in_vector_ccmp(vld1q_u64((const uint64_t*)(buf.get() + i + 16)));
      benchmark::DoNotOptimize(buf.get());
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_IsAsciiCCmp);

bool only_ascii_in_vector_ccmp_lane(uint64x2_t input)
{
    uint64x2_t and_result = vandq_u64(input, vdupq_n_u64(0x8080808080808080));
    uint64_t lo = vgetq_lane_u64(and_result, 0);
    uint64_t hi = vgetq_lane_u64(and_result, 1);
    return lo != 0 ? false : hi == 0;
}

static void BM_IsAsciiCCmp_lane(benchmark::State& state) {
  std::unique_ptr<unsigned char[]> buf(new unsigned char[1024 * 1024]);
  for (size_t i = 0; i < 1024*1024; ++i) {
    buf[i] = 'a';
  }

  for (auto _ : state) {
    bool res = false;
    for (size_t i = 0; i < 1024*1024; i += 32) {
      res &= only_ascii_in_vector_ccmp_lane(vld1q_u64((const uint64_t*)(buf.get() + i)));
      res &= only_ascii_in_vector_ccmp_lane(vld1q_u64((const uint64_t*)(buf.get() + i + 16)));
      benchmark::DoNotOptimize(buf.get());
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_IsAsciiCCmp_lane);

bool only_ascii_in_vector_ccmp_lane_2(uint64x2_t input)
{
    uint64x2_t and_result = vandq_u64(input, vdupq_n_u64(0x8080808080808080));
    uint64_t lo = vgetq_lane_u64(and_result, 0);
    if (lo != 0) {
        return false;
    }
    return vgetq_lane_u64(and_result, 1) == 0;
}

static void BM_IsAsciiCCmp_lane_2(benchmark::State& state) {
  std::unique_ptr<unsigned char[]> buf(new unsigned char[1024 * 1024]);
  for (size_t i = 0; i < 1024*1024; ++i) {
    buf[i] = 'a';
  }

  for (auto _ : state) {
    bool res = false;
    for (size_t i = 0; i < 1024*1024; i += 32) {
      res &= only_ascii_in_vector_ccmp_lane_2(vld1q_u64((const uint64_t*)(buf.get() + i)));
      res &= only_ascii_in_vector_ccmp_lane_2(vld1q_u64((const uint64_t*)(buf.get() + i + 16)));
      benchmark::DoNotOptimize(buf.get());
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_IsAsciiCCmp_lane_2);

bool only_ascii_in_vector_ccmp_8x16(uint8x16_t input)
{
    uint8x16_t and_result = vandq_u8(input, vdupq_n_u8(0x80));
    uint8x8_t lo = vget_low_u8(and_result);
    uint8x8_t hi = vget_high_u8(and_result);
    return (uint64_t)lo != 0 ? false : (uint64_t)hi == 0;
}

static void BM_IsAsciiCCmp_8x16(benchmark::State& state) {
  std::unique_ptr<unsigned char[]> buf(new unsigned char[1024 * 1024]);
  for (size_t i = 0; i < 1024*1024; ++i) {
    buf[i] = 'a';
  }

  for (auto _ : state) {
    bool res = false;
    for (size_t i = 0; i < 1024*1024; i += 32) {
      res &= only_ascii_in_vector_ccmp_8x16(vld1q_u8(buf.get() + i));
      res &= only_ascii_in_vector_ccmp_8x16(vld1q_u8(buf.get() + i + 16));
      benchmark::DoNotOptimize(buf.get());
      benchmark::DoNotOptimize(res);
    }
  }
}
BENCHMARK(BM_IsAsciiCCmp_8x16);

BENCHMARK_MAIN();
