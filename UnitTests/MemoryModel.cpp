#include "pch.h"
#include "MemoryModel.h"

void MemoryModel::Memcpy(const std::vector<ByteArray>& source_vec, const void* source,
	std::vector<ByteArray>& destination_vec, void* destination,
	std::size_t count)
{
	auto src = std::find_if(source_vec.begin(), source_vec.end(),
		[source](const auto& arr) { return reinterpret_cast<const void*>(arr.data()) == source; });
	auto dst = std::find_if(destination_vec.begin(), destination_vec.end(),
		[destination](const auto& arr) { return reinterpret_cast<const void*>(arr.data()) == destination; });

	if ((src != source_vec.end()) && (dst != destination_vec.end())) // source and destination are within the cuda memory model.
		std::copy_n(src->begin(), count, dst->begin());
	else // It's memory not from within the cuda memory model
		std::copy_n(reinterpret_cast<const uint8_t *>(source), count, reinterpret_cast<uint8_t*>(destination));
}

void MemoryModel::clear()
{
	host.clear();
	device.clear();
}

bool MemoryModel::empty() const
{
	return host.empty() && device.empty();
}

void* MemoryModel::MallocHost(std::size_t size)
{
	host.emplace_back(ByteArray(size));
	return reinterpret_cast<void*>(host.back().data());
}

void MemoryModel::FreeHost(void* p)
{
	const ByteArray::pointer pointer = reinterpret_cast<const ByteArray::pointer>(p);

	const auto it = std::remove_if(host.begin(), host.end(),
		[pointer](const auto& arr) { return arr.data() == pointer; });
	host.erase(it, host.end());
}

void* MemoryModel::MallocDevice(std::size_t size)
{
	device.emplace_back(ByteArray(size));
	return reinterpret_cast<void*>(device.back().data());
}

void MemoryModel::FreeDevice(void* p)
{
	const ByteArray::pointer pointer = reinterpret_cast<const ByteArray::pointer>(p);

	const auto it = std::remove_if(device.begin(), device.end(),
		[pointer](auto&& arr) { return arr.data() == pointer; });
	device.erase(it, device.end());
}

void MemoryModel::MemcpyHostToHost(const void* source, void* destination, std::size_t count)
{
	Memcpy(host, source, host, destination, count);
}

void MemoryModel::MemcpyHostToDevice(const void* source, void* destination, std::size_t count)
{
	Memcpy(host, source, device, destination, count);
}

void MemoryModel::MemcpyDeviceToHost(const void* source, void* destination, std::size_t count)
{
	Memcpy(device, source, host, destination, count);
}

void MemoryModel::MemcpyDeviceToDevice(const void* source, void* destination, std::size_t count)
{
	Memcpy(device, source, device, destination, count);
}
