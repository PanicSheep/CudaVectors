#include "pch.h"
#include "CudaMock.h"
#include <type_traits>

class HostVector_of_int : public CudaMock
{};

TEST(HostVector_of_int, is_default_constructible)
{
	ASSERT_TRUE(std::is_default_constructible_v<HostVector<int>>);
}

TEST(HostVector_of_int, is_copy_constructible)
{
	ASSERT_TRUE(std::is_copy_constructible_v<HostVector<int>>);
}

TEST(HostVector_of_int, is_copy_assignable)
{
	ASSERT_TRUE(std::is_copy_assignable_v<HostVector<int>>);
}

TEST(HostVector_of_int, is_move_constructible)
{
	ASSERT_TRUE(std::is_move_constructible_v<HostVector<int>>);
}

TEST(HostVector_of_int, is_move_assignable)
{
	ASSERT_TRUE(std::is_move_assignable_v<HostVector<int>>);
}

TEST(HostVector_of_int, destructor_frees_memory)
{
	HostVector<int> tmp(1);
	tmp.~HostVector();
}

TEST(HostVector_of_int, stores_std_vector)
{
	std::vector<int> vec{ 1,2,3 }; // arbitrary
	HostVector<int> host(vec);
	ASSERT_TRUE(vec == host.load());
}