/**
 * 1004 - Max Consecutive Ones III
 * Sliding Window
 * Time: O(n) | Space: O(1)
 * @param {number[]} nums
 * @param {number} k
 * @return {number}
 */
const longestOnes = function (nums, k) {
  let zeros = 0;
  let res = 0;
  let left = 0;

  for (let right = 0; right < nums.length; right++) {
    if (nums[right] === 0) {
      zeros++;
    }

    // Loop until at most k zeros
    while (zeros > k) {
      if (nums[left] === 0) {
        zeros--;
      }
      left++;
    }

    res = Math.max(res, right - left + 1);
  }

  return res;
};
