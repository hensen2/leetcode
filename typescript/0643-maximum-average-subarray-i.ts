/**
 * 643 - Maximum Average Subarray I
 * Sliding window | Fixed size window
 * Time: O(n) | Space: O(1)
 */

function findMaxAverage(nums: number[], k: number): number {
  let curr = 0;

  for (let i = 0; i < k; i++) {
    curr += nums[i];
  }

  let res = curr / k;

  for (let i = k; i < nums.length; i++) {
    curr += nums[i] - nums[i - k];
    res = Math.max(res, curr / k);
  }

  return res;
}
