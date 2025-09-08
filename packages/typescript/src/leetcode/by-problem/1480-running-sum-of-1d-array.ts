/**
 * 1480 - Running Sum of 1D Array
 * Prefix Sum
 * Time: O(n) | Space: O(1)
 */

function runningSum(nums: number[]): number[] {
  for (let i = 1; i < nums.length; i++) {
    nums[i] += nums[i - 1];
  }

  return nums;
}
