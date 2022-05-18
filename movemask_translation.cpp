#include <arm_neon.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>

#include "benchmark/benchmark.h"



uint32_t MoveMaskSSE2NEON(uint8x16_t cmp_res) {
  uint16x8_t high_bits = vreinterpretq_u16_u8(vshrq_n_u8(cmp_res, 7));
  uint32x4_t paired16 =
      vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 7));
  uint64x2_t paired32 =
      vreinterpretq_u64_u32(vsraq_n_u32(paired16, paired16, 14));
  uint8x16_t paired64 =
      vreinterpretq_u8_u64(vsraq_n_u64(paired32, paired32, 28));
  uint32_t mask =
      vgetq_lane_u8(paired64, 0) | ((int)vgetq_lane_u8(paired64, 8) << 8);
  return mask;
}

__attribute__((noinline)) uint64_t Get64BitSSE2NeonMask(const unsigned char* src, char ch) {
  const uint8x16_t dup = vdupq_n_u8(ch);
  uint64_t x0 = MoveMaskSSE2NEON(vceqq_u8(vld1q_u8(src), dup));
  uint64_t x1 = MoveMaskSSE2NEON(vceqq_u8(vld1q_u8(src + 16), dup));
  uint64_t x2 = MoveMaskSSE2NEON(vceqq_u8(vld1q_u8(src + 32), dup));
  uint64_t x3 = MoveMaskSSE2NEON(vceqq_u8(vld1q_u8(src + 48), dup));
  return x0 | (x1 << 16) | (x2 << 32) | (x3 << 48);
}

__attribute__((noinline)) uint64_t Get64BitZSTD(const unsigned char* src, char ch) {
  const uint8x16x4_t chunk = vld4q_u8(src);
  const uint8x16_t dup = vdupq_n_u8(ch);
  const uint8x16_t cmp0 = vceqq_u8(chunk.val[0], dup);
  const uint8x16_t cmp1 = vceqq_u8(chunk.val[1], dup);
  const uint8x16_t cmp2 = vceqq_u8(chunk.val[2], dup);
  const uint8x16_t cmp3 = vceqq_u8(chunk.val[3], dup);

  const uint8x16_t t0 = vsriq_n_u8(cmp1, cmp0, 1);
  const uint8x16_t t1 = vsriq_n_u8(cmp3, cmp2, 1);
  const uint8x16_t t2 = vsriq_n_u8(t1, t0, 2);
  const uint8x16_t t3 = vsriq_n_u8(t2, t2, 4);
  const uint8x8_t t4 = vshrn_n_u16(vreinterpretq_u16_u8(t3), 4);
  return vget_lane_u64(vreinterpret_u64_u8(t4), 0);
}

__attribute__((noinline)) uint64_t Get64BitGeoffLangdale(const unsigned char* src, char ch) {
  const uint8x16_t bitmask1 = { 0x01, 0x10, 0x01, 0x10, 0x01, 0x10, 0x01, 0x10,
                                0x01, 0x10, 0x01, 0x10, 0x01, 0x10, 0x01, 0x10};
  const uint8x16_t bitmask2 = { 0x02, 0x20, 0x02, 0x20, 0x02, 0x20, 0x02, 0x20,
                                0x02, 0x20, 0x02, 0x20, 0x02, 0x20, 0x02, 0x20};
  const uint8x16_t bitmask3 = { 0x04, 0x40, 0x04, 0x40, 0x04, 0x40, 0x04, 0x40,
                                0x04, 0x40, 0x04, 0x40, 0x04, 0x40, 0x04, 0x40};
  const uint8x16_t bitmask4 = { 0x08, 0x80, 0x08, 0x80, 0x08, 0x80, 0x08, 0x80,
                                0x08, 0x80, 0x08, 0x80, 0x08, 0x80, 0x08, 0x80};
  const uint8x16x4_t chunk = vld4q_u8(src);
  const uint8x16_t dup = vdupq_n_u8(ch);
  const uint8x16_t cmp0 = vceqq_u8(chunk.val[0], dup);
  const uint8x16_t cmp1 = vceqq_u8(chunk.val[1], dup);
  const uint8x16_t cmp2 = vceqq_u8(chunk.val[2], dup);
  const uint8x16_t cmp3 = vceqq_u8(chunk.val[3], dup);

  uint8x16_t t0 = vandq_u8(cmp0, bitmask1);
  uint8x16_t t1 = vbslq_u8(bitmask2, cmp1, t0);
  uint8x16_t t2 = vbslq_u8(bitmask3, cmp2, t1);
  uint8x16_t tmp = vbslq_u8(bitmask4, cmp3, t2);
  uint8x16_t sum = vpaddq_u8(tmp, tmp);
  return vgetq_lane_u64(vreinterpretq_u64_u8(sum), 0);
}

uint64_t _mm_movemask_aarch64(uint8x16_t input) {
    int8_t const ucShift[] =
    {
        -7, -6, -5, -4, -3, -2, -1, 0, -7, -6, -5, -4, -3, -2, -1, 0
    };
    int8x16_t vshift = vld1q_s8(ucShift);
    uint8x16_t vmask = vandq_u8(input, vdupq_n_u8(0x80));    
    vmask = vshlq_u8(vmask, vshift);
    uint64_t result;
    result = vaddv_u8(vget_low_u8(vmask));
    result += (vaddv_u8(vget_high_u8(vmask)) << 8);
    
    return result;
}

__attribute__((noinline)) uint64_t Get64BitNaive(const unsigned char* src, char ch) {
  const uint8x16_t dup = vdupq_n_u8(ch);
  uint64_t x0 = _mm_movemask_aarch64(vceqq_u8(vld1q_u8(src), dup));
  uint64_t x1 = _mm_movemask_aarch64(vceqq_u8(vld1q_u8(src + 16), dup));
  uint64_t x2 = _mm_movemask_aarch64(vceqq_u8(vld1q_u8(src + 32), dup));
  uint64_t x3 = _mm_movemask_aarch64(vceqq_u8(vld1q_u8(src + 48), dup));
  return x0 | (x1 << 16) | (x2 << 32) | (x3 << 48);
}

uint64_t _mm_movemask_aarch64_reduced(uint8x16_t input) {
    const uint8x16_t bitmask = {0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
                               0x01, 0x02, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
    uint8x16_t vmask = vandq_u8(input, bitmask);

    uint64_t result;
    result = vaddv_u8(vget_low_u8(vmask));
    result += (vaddv_u8(vget_high_u8(vmask)) << 8);
    
    return result;
}

__attribute__((noinline)) uint64_t Get64BitNaiveReduced(const unsigned char* src, char ch) {
  const uint8x16_t dup = vdupq_n_u8(ch);
  uint64_t x0 = _mm_movemask_aarch64_reduced(vceqq_u8(vld1q_u8(src), dup));
  uint64_t x1 = _mm_movemask_aarch64_reduced(vceqq_u8(vld1q_u8(src + 16), dup));
  uint64_t x2 = _mm_movemask_aarch64_reduced(vceqq_u8(vld1q_u8(src + 32), dup));
  uint64_t x3 = _mm_movemask_aarch64_reduced(vceqq_u8(vld1q_u8(src + 48), dup));
  return x0 | (x1 << 16) | (x2 << 32) | (x3 << 48);
}

static constexpr int kBufSize = 1 << 20;

std::unique_ptr<unsigned char[]> FillBuffer() {
  std::unique_ptr<unsigned char[]> buf(new unsigned char[kBufSize]);
  for (size_t i = 0; i < kBufSize / 2; ++i) {
    buf[i] = 'a';
    if (i % 23 == 0) {
      buf[i] = 'b';
    }
    if (i % 43 == 0) {
      buf[i] = 'b';
    }
  }
  for (size_t i = kBufSize / 2; i < kBufSize; ++i) {
    buf[i] = 'b';
  }
  return buf;
}

static void BM_Movemask64byteSSE2NEON(benchmark::State& state) {
  auto buf = FillBuffer();
  for (auto _ : state) {
    for (size_t i = 0; i + 64 < kBufSize; i += 64) {
      benchmark::DoNotOptimize(Get64BitSSE2NeonMask(buf.get() + i, 'a'));
    }
  }
}
BENCHMARK(BM_Movemask64byteSSE2NEON);

static void BM_Movemask64byteZSTD(benchmark::State& state) {
  auto buf = FillBuffer();
  for (auto _ : state) {
    for (size_t i = 0; i + 64 < kBufSize; i += 64) {
      benchmark::DoNotOptimize(Get64BitZSTD(buf.get() + i, 'a'));
    }
  }
}
BENCHMARK(BM_Movemask64byteZSTD);

static void BM_Movemask64byteGeoffrey(benchmark::State& state) {
  auto buf = FillBuffer();
  for (auto _ : state) {
    for (size_t i = 0; i + 64 < kBufSize; i += 64) {
      benchmark::DoNotOptimize(Get64BitGeoffLangdale(buf.get() + i, 'a'));
    }
  }
}
BENCHMARK(BM_Movemask64byteGeoffrey);

static void BM_Get64BitNaive(benchmark::State& state) {
  auto buf = FillBuffer();
  for (auto _ : state) {
    for (size_t i = 0; i + 64 < kBufSize; i += 64) {
      benchmark::DoNotOptimize(Get64BitNaive(buf.get() + i, 'a'));
    }
  }
}
BENCHMARK(BM_Get64BitNaive);

static void BM_Get64BitNaiveReduced(benchmark::State& state) {
  auto buf = FillBuffer();
  for (auto _ : state) {
    for (size_t i = 0; i + 64 < kBufSize; i += 64) {
      benchmark::DoNotOptimize(Get64BitNaiveReduced(buf.get() + i, 'a'));
    }
  }
}
BENCHMARK(BM_Get64BitNaiveReduced);

BENCHMARK_MAIN();
