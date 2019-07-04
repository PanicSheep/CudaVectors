#include "pch.h"
#include "MemoryModel.h"

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
	auto it = std::remove_if(host.begin(), host.end(), 
		[p](auto&& arr) { return reinterpret_cast<void*>(arr.data()) == p; });
	host.erase(it, host.end());
}