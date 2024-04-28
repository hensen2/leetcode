/*
    1426 - Counting Elements
    Hashset | Check existence
    Time: O(n) | Space: O(n)
*/

function countElements(arr: number[]): number {
  let res = 0;
  const set: Set<number> = new Set(arr);

  for (const i of arr) {
    if (set.has(i + 1)) {
      res++;
    }
  }

  return res;
}
