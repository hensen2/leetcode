/**
 * 125 - Valid Palindrome
 * Two pointers | Middle convergance
 * Time: O(n) | Space: O(1)
 * @param {string} s
 * @return {boolean}
 */
const isPalindrome = function (s) {
  let l = 0;
  let r = s.length - 1;

  while (l < r) {
    if (s[l] != s[r]) {
      return false;
    }

    l++;
    r--;
  }

  return true;
};
