/**
 * 1133 - Largest Unique Number
 * Hashmap | Count frequencies
 * Time: O(n) | Space: O(n)
 * @param {number[]} nums
 * @return {number}
 */
const largestUniqueNumber = function (nums) {
  const counts = new Map();
  let res = -1;

  for (const num of nums) {
    counts.set(num, (counts.get(num) || 0) + 1);
  }

  for (const [num, count] of counts) {
    if (count === 1) {
      res = Math.max(res, num);
    }
  }

  return res;
};
