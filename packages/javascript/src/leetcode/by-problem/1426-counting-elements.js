/**
 *  1426 - Counting Elements
 * Hashset | Check existence
 * Time: O(n) | Space: O(n)
 * @param {number[]} arr
 * @return {number}
 */
const countElements = function (arr) {
  let res = 0;
  const set = new Set(arr);

  for (const i of arr) {
    if (set.has(i + 1)) {
      res++;
    }
  }

  return res;
};
