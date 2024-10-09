/*
    344 - Reverse String
    Two pointers | Middle Convergence
    Time: O(n) | Space: O(1)
    Do not return anything, modify s in-place instead.
*/

#include <vector>
#include <utility>

class Solution
{
public:
    void reverseString(std::vector<char> &s)
    {
        int l = 0;
        int r = s.size() - 1;

        while (l < r)
        {
            std::swap(s[l], s[r]);

            l++;
            r--;
        }
    }
};