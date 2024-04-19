/**
 * @param {string} s
 * @return {boolean}
 */
const checkIfPalindrome = function (s) {
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

/*
    Time complexity: O(n)
    Space complexity: O(1)
*/
