#pragma once
#include <memory>
#include <vector>

class MemoryModel
{
	using ByteArray = std::vector<uint8_t>;
	std::vector<ByteArray> host, device;

	void Memcpy(const std::vector<ByteArray>& source_vec, const void* source,
		std::vector<ByteArray>& destination_vec, void* destination,
		std::size_t count);
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
	void FreeHost(void*);

	void* MallocDevice(std::size_t size);
	void FreeDevice(void*);

	void MemcpyHostToHost(const void* source, void* destination, std::size_t count);
	void MemcpyHostToDevice(const void* source, void* destination, std::size_t count);
	void MemcpyDeviceToHost(const void* source, void* destination, std::size_t count);
	void MemcpyDeviceToDevice(const void* source, void* destination, std::size_t count);
};