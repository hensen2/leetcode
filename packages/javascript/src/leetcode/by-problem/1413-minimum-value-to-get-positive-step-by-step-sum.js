/**
 * 1413 - Minimum Value to Get Positive Step by Step Sum
 * Prefix Sum | Greedy
 * Time: O(n) | Space: O(1)
 * @param {number[]} nums
 * @return {number}
 */
const minStartValue = function (nums) {
  let curr = 0;
  let minSum = 0;

  for (let i = 0; i < nums.length; i++) {
    curr += nums[i];
    minSum = Math.min(minSum, curr);
  }

  return 1 - minSum; // minSum + x = 1
};
