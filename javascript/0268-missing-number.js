/**
 * 268 - Missing Number
 * Hashset | Check existence
 * Time: O(n) | Space: O(n)
 * @param {number[]} nums
 * @return {number}
 */
const missingNumber = function (nums) {
  const n = nums.length;
  const set = new Set(nums);

  for (let i = 0; i < n; i++) {
    if (!set.has(i)) {
      return i;
    }
  }

  return n;
};
