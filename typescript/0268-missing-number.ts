/*
    268 - Missing Number
    Hashset | Check existence
    Time: O(n) | Space: O(n)
*/

function missingNumber(nums: number[]): number {
  const n = nums.length;
  const set: Set<number> = new Set(nums);

  for (let i = 0; i < n; i++) {
    if (!set.has(i)) {
      return i;
    }
  }

  return n;
}
