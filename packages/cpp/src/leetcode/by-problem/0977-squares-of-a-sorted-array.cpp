/*
    977 - Squares of a Sorted Array
    Two pointers | Iterate backwards
    Time: O(n) | Space: O(1)
*/

#include <vector>

class Solution
{
public:
    std::vector<int> sortedSquares(std::vector<int> &nums)
    {
        int n = nums.size();
        int l = 0, r = n - 1;
        std::vector<int> ans(n);

        for (int i = n - 1; i >= 0; i--)
        {
            int left = abs(nums[l]), right = abs(nums[r]);

            if (left > right)
            {
                ans[i] = pow(left, 2);
                l++;
            }
            else
            {
                ans[i] = pow(right, 2);
                r--;
            }
        }
        return ans;
    }
};