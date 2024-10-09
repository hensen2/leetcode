/**
 * 344 - Reverse String
 * Two pointers | Middle Convergence
 * Time: O(n) | Space: O(1)
 * @param {character[]} s
 * @return {void} Do not return anything, modify s in-place instead.
 */
const reverseString = function (s) {
  let l = 0;
  let r = s.length - 1;

  while (l < r) {
    let left = s[l];
    let right = s[r];
    s[l] = right;
    s[r] = left;

    l++;
    r--;
  }
};
