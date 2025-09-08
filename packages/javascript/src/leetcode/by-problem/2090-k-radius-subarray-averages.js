/**
 * 2090 - K Radius Subarray Averages
 * Sliding Window | Fixed size window
 * Time: O(n) | Space: O(1)
 * @param {number[]} nums
 * @param {number} k
 * @return {number[]}
 */
const getAverages = function (nums, k) {
  // Edge case when k = 0, its average will be itself
  if (k === 0) {
    return nums;
  }

  const windowSize = 2 * k + 1;
  const n = nums.length;
  const avgs = new Array(n).fill(-1);

  // Edge case when any index won't have k elements on each side
  if (windowSize > n) {
    return avgs;
  }

  // First get sum of first window of nums
  let windowSum = 0;
  for (let i = 0; i < windowSize; i++) {
    windowSum += nums[i];
  }
  // The first k-radius center
  avgs[k] = Math.floor(windowSum / windowSize);

  // Iterate from windowSize to n using sliding window method to update windowSum
  for (let i = windowSize; i < n; i++) {
    windowSum += nums[i] - nums[i - windowSize];
    avgs[i - k] = Math.floor(windowSum / windowSize);
  }

  return avgs;
};
