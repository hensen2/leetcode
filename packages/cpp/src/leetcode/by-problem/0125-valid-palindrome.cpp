/*
    125 - Valid Palindrome
    Two pointers | Middle Convergence
    Time: O(n) | Space: O(1)
*/

#include <string>

class Solution
{
public:
    bool isPalindrome(std::string s)
    {
        int l = 0;
        int r = s.size() - 1;

        while (l < r)
        {
            if (s[l] != s[r])
            {
                return false;
            }
            l++;
            r--;
        }

        return true;
    }
};
