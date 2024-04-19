/**
 * 125 - Valid Palindrome
 * Two pointers | Middle convergance
 * Time: O(n) | Space: O(1)
 * @param {string} s
 * @return {boolean}
 */
const isPalindrome = function (s) {
  let left = 0;
  let right = s.length - 1;

  while (left < right) {
    if (s[left] != s[right]) {
      return false;
    }

    left++;
    right--;
  }

  return true;
};
