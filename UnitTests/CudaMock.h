#pragma once
#include "pch.h"
#include "MemoryModel.h"
#include <vector>

static MemoryModel cuda_memory_model;

enum cudaError_t;
enum cudaMemcpyKind;

cudaError_t cudaMallocHost(void** ptr, std::size_t size);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind);

template <typename T>
cudaError_t cudaMallocHost(T** ptr, std::size_t size) { return cudaMallocHost((void**)ptr, size); }

template <typename T>
cudaError_t cudaFreeHost(T* ptr) { return cudaFreeHost((void*)ptr); }


class CudaMock : public ::testing::Test
{
protected:

	void SetUp() final;
	void TearDown() final;
};
