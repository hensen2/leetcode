/**
 * 643 - Maximum Average Subarray I
 * Sliding window | Fixed size window
 * Time: O(n) | Space: O(1)
 * @param {number[]} nums
 * @param {number} k
 * @return {number}
 */
const findMaxAverage = function (nums, k) {
  let curr = 0;

  for (let i = 0; i < k; i++) {
    curr += nums[i];
  }

  let res = curr;

  for (let i = k; i < nums.length; i++) {
    curr += nums[i] - nums[i - k];
    res = Math.max(res, curr);
  }

  return res / k;
};
