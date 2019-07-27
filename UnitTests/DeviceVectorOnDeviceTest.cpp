#include "pch.h"
#include "CudaMock.h"

namespace device
{
	class DeviceVector_of_int_from_device : public CudaMock
	{};

	TEST_F(DeviceVector_of_int_from_device, is_not_default_constructible)
	{
		ASSERT_FALSE(std::is_default_constructible_v<DeviceVector<int>>);
	}

	TEST_F(DeviceVector_of_int_from_device, is_copy_constructible)
	{
		ASSERT_TRUE(std::is_copy_constructible_v<DeviceVector<int>>);
	}

	TEST_F(DeviceVector_of_int_from_device, is_copy_assignable)
	{
		ASSERT_TRUE(std::is_copy_assignable_v<DeviceVector<int>>);
	}

	TEST_F(DeviceVector_of_int_from_device, is_move_constructible)
	{
		ASSERT_TRUE(std::is_move_constructible_v<DeviceVector<int>>);
	}

	TEST_F(DeviceVector_of_int_from_device, is_move_assignable)
	{
		ASSERT_TRUE(std::is_move_assignable_v<DeviceVector<int>>);
	}
	
	TEST_F(DeviceVector_of_int_from_device, constructible_from_DeviceVector_on_host)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1,2,3 } }; // arbitrary
		DeviceVector<int> from_device{ from_host };
		ASSERT_EQ(from_device.size(), from_host.size());
		ASSERT_EQ(from_device.capacity(), from_host.capacity());
	}

	TEST_F(DeviceVector_of_int_from_device, index_based_acces)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device[1], 2);
	}

	TEST_F(DeviceVector_of_int_from_device, index_based_acces_when_const)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		const DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device[1], 2);
	}

	TEST_F(DeviceVector_of_int_from_device, at_acces)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary	
		ASSERT_EQ(device.at(1), 2);
	}

	TEST_F(DeviceVector_of_int_from_device, at_acces_when_const)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		const DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.at(1), 2);
	}

	TEST_F(DeviceVector_of_int_from_device, at_throws_error_upon_bad_index)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_THROW(device.at(3), std::out_of_range);
	}

	TEST_F(DeviceVector_of_int_from_device, at_throws_error_upon_bad_index_when_const)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		const DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_THROW(device.at(3), std::out_of_range);
	}

	TEST_F(DeviceVector_of_int_from_device, front_access)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.front(), 1);
	}

	TEST_F(DeviceVector_of_int_from_device, front_access_when_const)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		const DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.front(), 1);
	}

	TEST_F(DeviceVector_of_int_from_device, back_access)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.back(), 3);
	}

	TEST_F(DeviceVector_of_int_from_device, back_access_when_const)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		const DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.back(), 3);
	}

	TEST_F(DeviceVector_of_int_from_device, data_access)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.data(), &device.front());
	}

	TEST_F(DeviceVector_of_int_from_device, data_access_when_const)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		const DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.data(), &device.front());
	}

	TEST_F(DeviceVector_of_int_from_device, begin_points_to_front)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.begin(), &device.front());
	}

	TEST_F(DeviceVector_of_int_from_device, end_points_to_one_past_back)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.end(), device.begin() + device.size());
	}

	TEST_F(DeviceVector_of_int_from_device, end_points_to_one_past_back_when_const)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.end(), device.begin() + device.size());
	}

	TEST_F(DeviceVector_of_int_from_device, cend_points_to_one_past_back)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_EQ(device.cend(), device.begin() + device.size());
	}

	TEST_F(DeviceVector_of_int_from_device, empty)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{} };
		DeviceVector<int> device{ from_host };
		ASSERT_TRUE(device.empty());
	}

	TEST_F(DeviceVector_of_int_from_device, non_empty)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_FALSE(device.empty());
	}

	TEST_F(DeviceVector_of_int_from_device, size_is_zero_for_empty)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{} };
		DeviceVector<int> device{ from_host };
		ASSERT_TRUE(device.empty());
	}

	TEST_F(DeviceVector_of_int_from_device, size_is_non_zero_for_non_empty)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		ASSERT_FALSE(device.empty());
	}

	TEST_F(DeviceVector_of_int_from_device, clear_preserves_capacity)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		const auto old_capacity = device.capacity();

		device.clear();

		ASSERT_EQ(device.capacity(), old_capacity);
	}

	TEST_F(DeviceVector_of_int_from_device, clear_makes_empty)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		device.clear();
		ASSERT_TRUE(device.empty());
	}

	TEST_F(DeviceVector_of_int_from_device, push_back_uses_capacity)
	{
		const std::size_t capacity = 100; // aribtrary
		host::DeviceVector<int> from_host(capacity);
		DeviceVector<int> device(from_host);

		for (int i = 0; i < capacity; i++)
			device.push_back(i);

		ASSERT_EQ(device.capacity(), capacity);
	}

	TEST_F(DeviceVector_of_int_from_device, push_back_xvalue_uses_capacity)
	{
		struct point { double x, y, z; };
		const std::size_t capacity = 100; // aribtrary
		host::DeviceVector<point> from_host(capacity);
		DeviceVector<point> device(from_host);

		for (int i = 0; i < capacity; i++)
			device.push_back(std::move(point()));

		ASSERT_EQ(device.capacity(), capacity);
	}

	TEST_F(DeviceVector_of_int_from_device, pop_back_preserves_capacity)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		const auto old_capacity = device.capacity();

		device.pop_back();

		ASSERT_EQ(device.capacity(), old_capacity);
	}

	TEST_F(DeviceVector_of_int_from_device, pop_back_changes_size)
	{
		host::DeviceVector<int> from_host{ std::vector<int>{ 1, 2, 3 }};
		DeviceVector<int> device{ from_host }; // arbitrary
		const auto old_size = device.size();

		device.pop_back();

		ASSERT_EQ(device.size(), old_size - 1);
	}

	TEST_F(DeviceVector_of_int_from_device, swap_member_function)
	{
		host::DeviceVector<int> from_host1{ std::vector<int>{ 1, 2, 3 } }; // arbitrary
		host::DeviceVector<int> from_host2{ std::vector<int>{ 4, 5, 6 } }; // arbitrary
		DeviceVector<int> device1(from_host1);
		DeviceVector<int> device2(from_host2);

		device1.swap(device2);

		bool equal1 = std::equal(device1.begin(), device1.end(), from_host2.begin(), from_host2.end(),
			[](const auto& l, const auto& r) { return l == r; });
		bool equal2 = std::equal(device2.begin(), device2.end(), from_host1.begin(), from_host1.end(),
			[](const auto& l, const auto& r) { return l == r; });
		ASSERT_TRUE(equal1);
		ASSERT_TRUE(equal2);
	}

	TEST_F(DeviceVector_of_int_from_device, swap_free_function)
	{
		host::DeviceVector<int> from_host1{ std::vector<int>{ 1, 2, 3 } }; // arbitrary
		host::DeviceVector<int> from_host2{ std::vector<int>{ 4, 5, 6 } }; // arbitrary
		DeviceVector<int> device1(from_host1);
		DeviceVector<int> device2(from_host2);

		swap(device1, device2);

		bool equal1 = std::equal(device1.begin(), device1.end(), from_host2.begin(), from_host2.end(),
			[](const auto& l, const auto& r) { return l == r; });
		bool equal2 = std::equal(device2.begin(), device2.end(), from_host1.begin(), from_host1.end(),
			[](const auto& l, const auto& r) { return l == r; });
		ASSERT_TRUE(equal1);
		ASSERT_TRUE(equal2);
	}
}