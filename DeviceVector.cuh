#pragma once
#include "Chronosity.h"
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cassert>

template<typename> class HostVector;

namespace host
{
	template <typename T>
	class DeviceVector
	{
		T* m_vec = nullptr;
		std::size_t m_size = 0;
		std::size_t m_capacity = 0;
	public:
		DeviceVector(const DeviceVector<T>&) = delete;
		DeviceVector(DeviceVector<T>&&) noexcept = default;
		DeviceVector<T>& operator=(const DeviceVector<T>&) = delete;
		DeviceVector<T>& operator=(DeviceVector<T>&&) noexcept = default;
		~DeviceVector() { cudaFree(m_vec); }

		explicit DeviceVector(std::size_t capacity) : m_capacity(capacity) { cudaMalloc(&m_vec, capacity * sizeof(T)); }
		DeviceVector(const std::vector<T>& o) { store(o); }

		DeviceVector<T>& operator=(const std::vector<T>& o) { store(o); return *this; }
		DeviceVector<T>& operator=(const HostVector<T>& o) { store(o, syn); return *this; }

		void assign(const std::vector<T>&);
		void assign(const HostVector<T>&, chronosity);
		void assign(const DeviceVector<T>&);

		void store(const std::vector<T>&);
		void store(const HostVector<T>&, chronosity);
		void store(const DeviceVector<T>&);

		std::vector<T> load() const;

		      T* data()       noexcept { return m_vec; }
		const T* data() const noexcept { return m_vec; }

		      T* begin()       noexcept { return m_vec; }
		const T* begin() const noexcept { return m_vec; }
		const T* cbegin() const noexcept { return m_vec; }
		      T* end()       noexcept { return m_vec + m_size; }
		const T* end() const noexcept { return m_vec + m_size; }
		const T* cend() const noexcept { return m_vec + m_size; }

		bool empty() const noexcept { return m_size == 0; }
		std::size_t size() const noexcept { return m_size; }
		std::size_t capacity() const noexcept { return m_capacity; }
		void reserve(std::size_t new_capacity);

		void clear() noexcept { m_size = 0; }
		void resize(std::size_t count) { reserve(count); m_size = count; }
		void swap(DeviceVector<T>& o) noexcept { std::swap(m_vec, o.m_vec); std::swap(m_size, o.m_size); std::swap(m_capacity, o.m_capacity); }
	};

	template <typename T>
	inline void swap(DeviceVector<T>& lhs, DeviceVector<T>& rhs) noexcept(noexcept(lhs.swap(rhs)))
	{
		lhs.swap(rhs);
	}
}

namespace device
{
	template <typename T>
	class DeviceVector
	{
		T* m_vec = nullptr;
		std::size_t m_size = 0;
		std::size_t m_capacity = 0;
	public:
		__device__ DeviceVector(T* begin, std::size_t size) : m_vec(begin), m_size(size) {}
		DeviceVector(const DeviceVector<T>& o) = default;
		DeviceVector(host::DeviceVector<T>& o) : m_vec(o.data()), m_size(o.size()), m_capacity(o.capacity()) {}
		DeviceVector<T>& operator=(const DeviceVector<T>&) = default;

		__device__       T& operator[](std::size_t pos)       noexcept { return m_vec[pos]; }
		__device__ const T& operator[](std::size_t pos) const noexcept { return m_vec[pos]; }
		__device__       T& at(std::size_t pos)       noexcept(false);
		__device__ const T& at(std::size_t pos) const noexcept(false);
		__device__       T& front()       noexcept { return m_vec[0]; }
		__device__ const T& front() const noexcept { return m_vec[0]; }
		__device__       T& back()       noexcept { return m_vec[m_size - 1]; }
		__device__ const T& back() const noexcept { return m_vec[m_size - 1]; }

		__device__       T* begin()       noexcept { return m_vec; }
		__device__ const T* begin() const noexcept { return m_vec; }
		__device__ const T* cbegin() const noexcept { return m_vec; }
		__device__       T* end()       noexcept { return m_vec + m_size; }
		__device__ const T* end() const noexcept { return m_vec + m_size; }
		__device__ const T* cend() const noexcept { return m_vec + m_size; }

		__device__ bool empty() const noexcept { return m_size == 0; }
		__device__ std::size_t size() const noexcept { return m_size; }
		__device__ std::size_t capacity() const noexcept { return m_capacity; }

		__device__ void clear() noexcept { m_size = 0; }
		__device__ void push_back(const T&);
		__device__ void push_back(T&&);
		__device__ void pop_back() { m_size--; }
		__device__ void swap(DeviceVector<T>& o) noexcept { ::swap(m_vec, o.m_vec); ::swap(m_size, o.m_size); ::swap(m_capacity, o.m_capacity); }
	};

	template <typename T>
	__device__ inline void swap(DeviceVector<T>& lhs, DeviceVector<T>& rhs) noexcept(noexcept(lhs.swap(rhs)))
	{
		lhs.swap(rhs);
	}
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace host
{
	template <typename T>
	void DeviceVector<T>::assign(const std::vector<T>& src)
	{
		assert(m_capacity >= src.size());

		cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);

		m_size = src.size();
	}

	template <typename T>
	void DeviceVector<T>::assign(const HostVector<T>& src, chronosity chrono)
	{
		assert(m_capacity >= src.size());

		if /*constexpr*/ (chrono == chronosity::syn)
			cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
		else
			cudaMemcpyAsync(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);

		m_size = src.size();
	}

	template <typename T>
	void DeviceVector<T>::assign(const DeviceVector<T>& src)
	{
		assert(m_capacity >= src.size());

		cudaMemcpy(m_vec, src.m_vec, src.size() * sizeof(T), cudaMemcpyDeviceToDevice);

		m_size = src.size();
	}

	template <typename T>
	void DeviceVector<T>::store(const std::vector<T>& src)
	{
		if (m_capacity < src.size())
		{
			DeviceVector<T> new_vec{ src.size() };
			swap(new_vec);
		}
		assign(src);
	}

	template <typename T>
	void DeviceVector<T>::store(const HostVector<T>& src, chronosity chrono)
	{
		if (m_capacity < src.size())
		{
			DeviceVector<T> new_vec{ src.size() };
			swap(new_vec);
		}
		assign(src, chrono);
	}

	template <typename T>
	void DeviceVector<T>::store(const DeviceVector<T>& src)
	{
		if (m_capacity < src.size())
		{
			DeviceVector<T> new_vec{ src.size() };
			swap(new_vec);
		}
		assign(src);
	}

	template<typename T>
	std::vector<T> DeviceVector<T>::load() const
	{
		std::vector<T> ret(m_size);
		cudaMemcpy(ret.data(), m_vec, m_size * sizeof(T), cudaMemcpyDeviceToHost);
		return ret;
	}

	template <typename T>
	void DeviceVector<T>::reserve(const std::size_t new_capacity)
	{
		if (new_capacity > m_capacity)
		{
			DeviceVector<T> new_vec{ new_capacity };
			new_vec.assign(*this);
			swap(new_vec);
		}
	}
}


namespace device
{
	template <typename T>
	__device__ T& DeviceVector<T>::at(std::size_t pos)
	{
		if (pos >= size())
			throw std::out_of_range{ "vector index out of range" };
		return m_vec[pos];
	}

	template <typename T>
	__device__ const T& DeviceVector<T>::at(std::size_t pos) const
	{
		if (pos >= size())
			throw std::out_of_range{ "vector index out of range" };
		return m_vec[pos];
	}

	template <typename T>
	__device__ void DeviceVector<T>::push_back(const T& value)
	{
		if (m_size >= m_capacity)
			throw std::runtime_error{ "Not enough memory" };
		m_vec[m_size++] = value;
	}

	template <typename T>
	__device__ void DeviceVector<T>::push_back(T&& value)
	{
		if (m_size >= m_capacity)
			throw std::runtime_error{ "Not enough memory" };
		m_vec[m_size++] = std::move(value);
	}
}
