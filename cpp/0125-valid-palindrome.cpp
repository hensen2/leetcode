/*
    125 - Valid Palindrome
    Two pointers | Middle convergance
    Time: O(n) | Space: O(1)
*/

#include <string>

class Solution
{
public:
    bool isPalindrome(std::string s)
    {
        int left = 0;
        int right = s.size() - 1;

        while (left < right)
        {
            if (s[left] != s[right])
            {
                return false;
            }
            left++;
            right--;
        }

        return true;
    }
};
