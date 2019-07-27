#include "pch.h"
#include "CudaMock.h"

class HostVector_of_int : public CudaMock
{};

TEST_F(HostVector_of_int, is_default_constructible)
{
	ASSERT_TRUE(std::is_default_constructible_v<HostVector<int>>);
}

TEST_F(HostVector_of_int, is_copy_constructible)
{
	ASSERT_TRUE(std::is_copy_constructible_v<HostVector<int>>);
}

TEST_F(HostVector_of_int, is_copy_assignable)
{
	ASSERT_TRUE(std::is_copy_assignable_v<HostVector<int>>);
}

TEST_F(HostVector_of_int, is_move_constructible)
{
	ASSERT_TRUE(std::is_move_constructible_v<HostVector<int>>);
}

TEST_F(HostVector_of_int, is_move_assignable)
{
	ASSERT_TRUE(std::is_move_assignable_v<HostVector<int>>);
}

TEST_F(HostVector_of_int, constructing_with_capacity_sets_size_and_capacity)
{
	HostVector<int> host{ 5 }; // arbitrary
	ASSERT_EQ(host.size(), 0);
	ASSERT_EQ(host.capacity(), 5);
}

TEST_F(HostVector_of_int, destructor_frees_memory)
{
	HostVector<int> tmp(1);
	tmp.~HostVector();
}

TEST_F(HostVector_of_int, constructible_from_std_vector)
{
	std::vector<int> vec{ 1,2,3 }; // arbitrary
	HostVector<int> host(vec);
	ASSERT_EQ(vec, host.load());
}

TEST_F(HostVector_of_int, assignable_from_std_vector)
{
	std::vector<int> vec{ 1,2,3 }; // arbitrary
	HostVector<int> host;
	host = vec;
	ASSERT_EQ(vec, host.load());
}

TEST_F(HostVector_of_int, equal_comparable)
{
	std::vector<int> vec{ 1,2,3 }; // arbitrary
	HostVector<int> host1(vec);
	HostVector<int> host2(vec);
	HostVector<int> host3{ std::vector<int>{ 1,2,4 } }; // arbitrary	

	ASSERT_TRUE(host1 == host2);
	ASSERT_FALSE(host1 == host3);
}

TEST_F(HostVector_of_int, not_equal_comparable)
{
	std::vector<int> vec{ 1,2,3 }; // arbitrary
	HostVector<int> host1(vec);
	HostVector<int> host2(vec);
	HostVector<int> host3{ std::vector<int>{ 1,2,4 } }; // arbitrary	

	ASSERT_FALSE(host1 != host2);
	ASSERT_TRUE(host1 != host3);
}

TEST_F(HostVector_of_int, index_based_acces)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary	
	ASSERT_EQ(host[1], 2);
}

TEST_F(HostVector_of_int, index_based_acces_when_const)
{
	const HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host[1], 2);
}

TEST_F(HostVector_of_int, at_acces)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary	
	ASSERT_EQ(host.at(1), 2);
}

TEST_F(HostVector_of_int, at_acces_when_const)
{
	const HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.at(1), 2);
}

TEST_F(HostVector_of_int, at_throws_error_upon_bad_index)
{
	 HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	 ASSERT_THROW(host.at(3), std::out_of_range);
}

TEST_F(HostVector_of_int, at_throws_error_upon_bad_index_when_const)
{
	const HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_THROW(host.at(3), std::out_of_range);
}

TEST_F(HostVector_of_int, front_access)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.front(), 1);
}

TEST_F(HostVector_of_int, front_access_when_const)
{
	const HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.front(), 1);
}

TEST_F(HostVector_of_int, back_access)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.back(), 3);
}

TEST_F(HostVector_of_int, back_access_when_const)
{
	const HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.back(), 3);
}

TEST_F(HostVector_of_int, data_access)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.data(), &host.front());
}

TEST_F(HostVector_of_int, data_access_when_const)
{
	const HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.data(), &host.front());
}

TEST_F(HostVector_of_int, begin_points_to_front)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.begin(), &host.front());
}

TEST_F(HostVector_of_int, begin_points_to_front_when_const)
{
	const HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.begin(), &host.front());
}

TEST_F(HostVector_of_int, cbegin_points_to_front)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.cbegin(), &host.front());
}

TEST_F(HostVector_of_int, end_points_to_one_past_back)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.end(), &host.back() + 1);
}

TEST_F(HostVector_of_int, end_points_to_one_past_back_when_const)
{
	const HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.end(), &host.back() + 1);
}

TEST_F(HostVector_of_int, cend_points_to_one_past_back)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_EQ(host.cend(), &host.back() + 1);
}

TEST_F(HostVector_of_int, empty)
{
	const HostVector<int> host{ std::vector<int>{} }; // arbitrary
	ASSERT_TRUE(host.empty());
}

TEST_F(HostVector_of_int, non_empty)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_FALSE(host.empty());
}

TEST_F(HostVector_of_int, size_is_zero_for_empty)
{
	const HostVector<int> host{ std::vector<int>{} }; // arbitrary
	ASSERT_EQ(host.size(), 0);
}

TEST_F(HostVector_of_int, size_is_non_zero_for_non_empty)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	ASSERT_NE(host.size(), 0);
}

TEST_F(HostVector_of_int, reserve_changes_capacity)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	const int new_capacity = 100; // arbitrary

	host.reserve(new_capacity);

	ASSERT_EQ(host.capacity(), new_capacity);
}

TEST_F(HostVector_of_int, reserve_preserves_size)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	const auto old_size = host.size();
	const int new_capacity = 100; // arbitrary

	host.reserve(new_capacity);

	ASSERT_EQ(host.size(), old_size);
}

TEST_F(HostVector_of_int, clear_preserves_capacity)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	const auto old_capacity = host.capacity();

	host.clear();

	ASSERT_EQ(host.capacity(), old_capacity);
}

TEST_F(HostVector_of_int, clear_makes_empty)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	host.clear();
	ASSERT_TRUE(host.empty());
}

TEST_F(HostVector_of_int, push_back_grows_automatically)
{
	HostVector<int> host;
	for (int i = 0; i < 100; i++) // aribtrary
		host.push_back(i);
	ASSERT_FALSE(host.empty());
}

TEST_F(HostVector_of_int, push_back_uses_capacity)
{
	const std::size_t capacity = 100; // aribtrary
	HostVector<int> host(capacity);

	for (int i = 0; i < capacity; i++)
		host.push_back(i);

	ASSERT_EQ(host.capacity(), capacity);
}

TEST_F(HostVector_of_int, push_back_xvalue_grows_automatically)
{
	struct point { double x, y, z; };
	HostVector<point> host;

	for (int i = 0; i < 100; i++) // aribtrary
		host.push_back(std::move(point()));

	ASSERT_FALSE(host.empty());
}

TEST_F(HostVector_of_int, push_back_xvalue_uses_capacity)
{
	struct point { double x, y, z; };
	const std::size_t capacity = 100; // aribtrary
	HostVector<point> host(capacity);

	for (int i = 0; i < capacity; i++)
		host.push_back(std::move(point()));

	ASSERT_EQ(host.capacity(), capacity);
}

TEST_F(HostVector_of_int, pop_back_preserves_capacity)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	const auto old_capacity = host.capacity();

	host.pop_back();

	ASSERT_EQ(host.capacity(), old_capacity);
}

TEST_F(HostVector_of_int, pop_back_changes_size)
{
	HostVector<int> host{ std::vector<int>{ 1,2,3 } }; // arbitrary
	const auto old_size = host.size();

	host.pop_back();

	ASSERT_EQ(host.size(), old_size - 1);
}

TEST_F(HostVector_of_int, resize_grows_when_needed)
{
	HostVector<int> host(100); // arbitrary
	host.push_back(8);
	const auto old_size = host.size();
	const auto old_capacity = host.capacity();

	host.resize(old_capacity + 1);

	ASSERT_GT(host.size(), old_size);
	ASSERT_GT(host.capacity(), old_capacity);
}

TEST_F(HostVector_of_int, resize_preserves_capacity_when_possible)
{
	HostVector<int> host(100); // arbitrary
	host.push_back(8);
	const auto old_capacity = host.capacity();

	host.resize(old_capacity);

	ASSERT_EQ(host.capacity(), old_capacity);
}

TEST_F(HostVector_of_int, resize_resets_size)
{
	HostVector<int> host(100); // arbitrary
	host.push_back(8);
	const auto old_size = host.size();
	const auto new_size = 89; // arbitrary

	host.resize(new_size);

	ASSERT_EQ(host.size(), new_size);
}

TEST_F(HostVector_of_int, swap_member_function)
{
	const std::vector<int> vec1{ 1, 2, 3 }; // arbitrary
	const std::vector<int> vec2{ 4, 5, 6 }; // arbitrary
	HostVector<int> host1{ vec1 };
	HostVector<int> host2{ vec2 };

	host1.swap(host2);

	ASSERT_TRUE(host1 == vec2);
	ASSERT_TRUE(host2 == vec1);
}

TEST_F(HostVector_of_int, swap_free_function)
{
	const std::vector<int> vec1{ 1, 2, 3 }; // arbitrary
	const std::vector<int> vec2{ 4, 5, 6 }; // arbitrary
	HostVector<int> host1{ vec1 };
	HostVector<int> host2{ vec2 };

	swap(host1, host2);

	ASSERT_TRUE(host1 == vec2);
	ASSERT_TRUE(host2 == vec1);
}