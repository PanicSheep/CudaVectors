#include "pch.h"
#include "CudaMock.h"

namespace host
{
	class DeviceVector_of_int_on_host : public CudaMock
	{};

	TEST_F(DeviceVector_of_int_on_host, is_default_constructible)
	{
		ASSERT_TRUE(std::is_default_constructible_v<DeviceVector<int>>);
	}

	TEST_F(DeviceVector_of_int_on_host, is_not_copy_constructible)
	{
		ASSERT_FALSE(std::is_copy_constructible_v<DeviceVector<int>>);
	}

	TEST_F(DeviceVector_of_int_on_host, is_not_copy_assignable)
	{
		ASSERT_FALSE(std::is_copy_assignable_v<DeviceVector<int>>);
	}

	TEST_F(DeviceVector_of_int_on_host, is_move_constructible)
	{
		ASSERT_TRUE(std::is_move_constructible_v<DeviceVector<int>>);
	}

	TEST_F(DeviceVector_of_int_on_host, is_move_assignable)
	{
		ASSERT_TRUE(std::is_move_assignable_v<DeviceVector<int>>);
	}

	TEST_F(DeviceVector_of_int_on_host, constructing_with_capacity_sets_size_and_capacity)
	{
		DeviceVector<int> host{ 5 }; // arbitrary
		ASSERT_EQ(host.size(), 0);
		ASSERT_EQ(host.capacity(), 5);
	}

	TEST_F(DeviceVector_of_int_on_host, destructor_frees_memory)
	{
		DeviceVector<int> tmp(1);
		tmp.~DeviceVector();
	}

	TEST_F(DeviceVector_of_int_on_host, constructible_from_std_vector)
	{
		std::vector<int> vec{ 1,2,3 }; // arbitrary
		DeviceVector<int> device(vec);
		ASSERT_EQ(vec, device.load());
	}

	TEST_F(DeviceVector_of_int_on_host, assignable_from_std_vector)
	{
		std::vector<int> vec{ 1,2,3 }; // arbitrary
		DeviceVector<int> device;
		device = vec;
		ASSERT_EQ(vec, device.load());
	}
	
	TEST_F(DeviceVector_of_int_on_host, begin_points_to_data)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		ASSERT_EQ(device.begin(), device.data());
	}

	TEST_F(DeviceVector_of_int_on_host, begin_points_to_data_when_const)
	{
		const DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		ASSERT_EQ(device.begin(), device.data());
	}

	TEST_F(DeviceVector_of_int_on_host, cbegin_points_to_data)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		ASSERT_EQ(device.cbegin(), device.data());
	}

	TEST_F(DeviceVector_of_int_on_host, end_points_to_one_past_back)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		ASSERT_EQ(device.end(), device.begin() + device.size());
	}

	TEST_F(DeviceVector_of_int_on_host, end_points_to_one_past_back_when_const)
	{
		const DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		ASSERT_EQ(device.end(), device.begin() + device.size());
	}

	TEST_F(DeviceVector_of_int_on_host, cend_points_to_one_past_back)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		ASSERT_EQ(device.cend(), device.begin() + device.size());
	}

	TEST_F(DeviceVector_of_int_on_host, empty)
	{
		const DeviceVector<int> device{ std::vector<int>{} };
		ASSERT_TRUE(device.empty());
	}

	TEST_F(DeviceVector_of_int_on_host, non_empty)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		ASSERT_FALSE(device.empty());
	}

	TEST_F(DeviceVector_of_int_on_host, size_is_zero_for_empty)
	{
		const DeviceVector<int> device{ std::vector<int>{} }; // arbitrary
		ASSERT_TRUE(device.empty());
	}

	TEST_F(DeviceVector_of_int_on_host, size_is_non_zero_for_non_empty)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		ASSERT_FALSE(device.empty());
	}

	TEST_F(DeviceVector_of_int_on_host, reserve_changes_capacity)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		const int new_capacity = 100; // arbitrary

		device.reserve(new_capacity);

		ASSERT_EQ(device.capacity(), new_capacity);
	}

	TEST_F(DeviceVector_of_int_on_host, reserve_preserves_size)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		const auto old_size = device.size();
		const int new_capacity = 100; // arbitrary

		device.reserve(new_capacity);

		ASSERT_EQ(device.size(), old_size);
	}

	TEST_F(DeviceVector_of_int_on_host, clear_preserves_capacity)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		const auto old_capacity = device.capacity();

		device.clear();

		ASSERT_EQ(device.capacity(), old_capacity);
	}

	TEST_F(DeviceVector_of_int_on_host, clear_makes_empty)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		device.clear();
		ASSERT_TRUE(device.empty());
	}

	TEST_F(DeviceVector_of_int_on_host, resize_grows_when_needed)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		device.reserve(100); // arbitrary
		const auto old_size = device.size();
		const auto old_capacity = device.capacity();

		device.resize(old_capacity + 1);

		ASSERT_GT(device.size(), old_size);
		ASSERT_GT(device.capacity(), old_capacity);
	}

	TEST_F(DeviceVector_of_int_on_host, resize_preserves_capacity_when_possible)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		device.reserve(100); // arbitrary
		const auto old_capacity = device.capacity();

		device.resize(old_capacity);

		ASSERT_EQ(device.capacity(), old_capacity);
	}

	TEST_F(DeviceVector_of_int_on_host, resize_resets_size)
	{
		DeviceVector<int> device{ std::vector<int>{ 1,2,3 } }; // arbitrary
		device.reserve(100); // arbitrary
		const auto old_size = device.size();
		const auto new_size = 89; // arbitrary

		device.resize(new_size);

		ASSERT_EQ(device.size(), new_size);
	}

	TEST_F(DeviceVector_of_int_on_host, swap_member_function)
	{
		const std::vector<int> vec1{ 1, 2, 3 }; // arbitrary
		const std::vector<int> vec2{ 4, 5, 6 }; // arbitrary
		DeviceVector<int> device1{ vec1 };
		DeviceVector<int> device2{ vec2 };

		device1.swap(device2);

		ASSERT_TRUE(device1.load() == vec2);
		ASSERT_TRUE(device2.load() == vec1);
	}

	TEST_F(DeviceVector_of_int_on_host, swap_free_function)
	{
		const std::vector<int> vec1{ 1, 2, 3 }; // arbitrary
		const std::vector<int> vec2{ 4, 5, 6 }; // arbitrary
		DeviceVector<int> device1{ vec1 };
		DeviceVector<int> device2{ vec2 };

		swap(device1, device2);

		ASSERT_TRUE(device1.load() == vec2);
		ASSERT_TRUE(device2.load() == vec1);
	}
}