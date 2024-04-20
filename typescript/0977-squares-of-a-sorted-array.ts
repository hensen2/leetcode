/**
 * 977 - Squares of a Sorted Array
 * Two pointers | Middle convergance
 * Time: O(n) | Space: O(1)
 */

function sortedSquares(nums: number[]): number[] {
  let l = 0;
  let r = nums.length - 1;
  const res: number[] = new Array(nums.length);

  while (l <= r) {
    let left = nums[l] * nums[l];
    let right = nums[r] * nums[r];

    if (left > right) {
      res[r - l] = left;
      l++;
    } else {
      res[r - l] = right;
      r--;
    }
  }

  return res;
}
