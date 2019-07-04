#pragma once
#include <memory>
#include <vector>

class MemoryModel
{
	using ByteArray = std::vector<uint8_t>;
	std::vector<ByteArray> host, device;
public:
	MemoryModel() = default;
	MemoryModel(const MemoryModel&) = delete;
	MemoryModel(MemoryModel&&) = delete;
	MemoryModel& operator=(const MemoryModel&) = delete;
	MemoryModel& operator=(MemoryModel&&) = delete;
	~MemoryModel() = default;

	void clear();
	bool empty() const;

	void* MallocHost(std::size_t size);
	void FreeHost(void* p);
};