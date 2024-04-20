/**
 * 977 - Squares of a Sorted Array
 * Two pointers | Middle convergance
 * Time: O(n) | Space: O(1)
 * @param {number[]} nums
 * @return {number[]}
 */
const sortedSquares = function (nums) {
  let l = 0;
  let r = nums.length - 1;
  const res = new Array(nums.length);

  while (l <= r) {
    let left = Math.pow(nums[l], 2);
    let right = Math.pow(nums[r], 2);

    if (left > right) {
      res[r - l] = left;
      l++;
    } else {
      res[r - l] = right;
      r--;
    }
  }

  return res;
};
